#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Preprocessor for genome data before parsing into graph structures. This
script deals with data wrangling that does not necessarily fit into more defined
downstream modules, which includes downloading shared files, setting up
directories and symbolic linkes, parsing tissue-specific data, and adjusting
file formatting."""


from collections import defaultdict
import contextlib
import csv
import subprocess
from typing import Dict, List, Union

import pandas as pd

from constants import REGULATORY_ELEMENTS
from utils import check_and_symlink
from utils import dir_check_make
from utils import time_decorator

# import requests

NODETYPES_LOCAL: List[str] = [
    "cpgislands",
    "ctcfccre",
    "tss",
]  # local context filetypes


def _mirna_ref(ref_file: str) -> Dict[str, List[str]]:
    """Reference for miRNA target genes"""
    mirnaref = defaultdict(list)
    with open(ref_file) as f:
        for line in csv.reader(f, delimiter="\t"):
            mirnaref[line[3]].append(line[-1])
    return mirnaref


class GenomeDataPreprocessor:
    """Data preprocessor for dealing with differences in bed files.

    Attributes:
        experiment_name (str): The name of the experiment.
        interaction_types (List[str]): The types of interactions.
        nodes (List[str]): The nodes.
        regulatory (str): The regulatory element catalogue to use.
        working_directory (str): The working directory.
        params (Dict[str, Dict[str, str]]): The parameters.

    Methods
    ----------
    _run_cmd:
        Runs shell command via subprocess call.
    _make_directories:
        Make required directories for processing.
    _symlink_rawdata:
        Symlink the raw data to a folder within the directory.
    _download_shared_files:
        Download files used by multiple tissues.
    _add_tad_id:
        Add identification number to each TAD.
    _superenhancers:
        Simple parser to remove superenhancer bed unneeded info.
    _tf_binding_sites:
        Parse tissue-specific transcription factor binding sites.
    _merge_cpg:
        Merge individual CPGs with optional liftover.
    prepare_data_files:
        Main pipeline function.

    Examples:
    --------
    >>> preprocessObject = GenomeDataPreprocessor(
            experiment_name=experiment_params["experiment_name"],
            interaction_types=experiment_params["interaction_types"],
            nodes=nodes,
            regulatory=experiment_params["regulatory"],
            working_directory=experiment_params["working_directory"],
            params=tissue_params,
        )

    >>> preprocessObject.prepare_data_files()
    """

    def __init__(
        self,
        experiment_name: str,
        interaction_types: List[str],
        nodes: List[str],
        regulatory: str,
        working_directory: str,
        params: Dict[str, Dict[str, str]],
    ) -> None:
        """Initialize the class"""
        self.experiment_name = experiment_name
        self.interaction_types = interaction_types
        self.nodes = nodes
        self.regulatory = regulatory
        self.working_directory = working_directory

        self.dirs = params["dirs"]
        self.interaction = params["interaction"]
        self.methylation = params["methylation"]
        self.resources = params["resources"]
        self.references = params["references"]
        self.shared = params["local"]
        self.features = params["features"]
        self.tissue_specific_nodes = params["tissue_specific_nodes"]

        self.tissue = self.resources["tissue"]
        self.root_dir = self.dirs["root_dir"]
        self.shared_data_dir = f"{self.root_dir}/shared_data"
        self.reg_dir = f"{self.shared_data_dir}/regulatory_elements"
        self.tissue_dir = (
            f"{self.working_directory}/{self.experiment_name}/{self.tissue}"
        )
        self.data_dir = f"{self.root_dir}/raw_tissue_data/{self.tissue}"

        # make directories, link files, and download shared files if necessary
        self._make_directories()
        self._symlink_rawdata()
        # self._download_shared_files()

    def _make_directories(self) -> None:
        """Make directories for processing"""
        dir_check_make(self.tissue_dir)

        for directory in ["local", "interaction", "unprocessed"]:
            dir_check_make(f"{self.tissue_dir}/{directory}")

    def _run_cmd(self, cmd: str) -> None:
        """Simple wrapper for subprocess as options across this script are
        constant"""
        subprocess.run(cmd, stdout=None, shell=True)

    def _symlink_rawdata(self) -> None:
        """Make symlinks for tissue specific files in unprocessed folder"""
        for file in self.tissue_specific_nodes.values():
            check_and_symlink(
                dst=f"{self.tissue_dir}/unprocessed/{file}",
                src=f"{self.data_dir}/{file}",
                boolean=True,
            )

        interact_files = {
            # "circuits": f"{self.dirs['circuit_dir']}/{self.interaction['circuits']}",
            "ppis": f"{self.shared_data_dir}/interaction/{self.interaction['ppis']}",
            "tf_marker": f"{self.shared_data_dir}/interaction/{self.interaction['tf_marker']}",
            "tf_binding": f"{self.shared_data_dir}/interaction/{self.interaction['tf_binding']}",
        }

        with contextlib.suppress(TypeError):
            for datatype in self.interaction_types:
                if datatype == "mirna":
                    check_and_symlink(
                        src=f"{self.shared_data_dir}/interaction/{self.interaction['mirnatargets']}",
                        dst=f"{self.tissue_dir}/interaction/"
                        + self.interaction["mirnatargets"],
                        boolean=True,
                    )
                else:
                    check_and_symlink(
                        src=interact_files[datatype],
                        dst=f"{self.tissue_dir}/interaction/{self.interaction[datatype]}",
                        boolean=False,
                    )

    # def _download_shared_files(self) -> None:
    #     """Download shared local features if not already present"""

    #     def download(url: str, filename: str) -> None:
    #         with open(filename, "wb") as file:
    #             response = requests.get(url)
    #             file.write(response.content)

    #     if os.listdir(f"{self.root_dir}/local_feats") != self.shared:
    #         for file in self.shared.values():
    #             download(
    #                 f"https://raw.github.com/sciencesteveho/genome_graph_perturbation/raw/master/shared_files/local_feats/{file}",
    #                 f"{self.root_dir}/shared_data/local_feats/{file}",
    #             )

    @time_decorator(print_args=True)
    def _add_tad_id(self, bed: str) -> None:
        """Add identification number to each TAD"""
        cmd = f"awk -v FS='\t' -v OFS='\t' '{{print $1, $2, $3, \"tad_\"NR}}' \
            {self.tissue_dir}/unprocessed/{bed} \
            > {self.tissue_dir}/local/tads_{self.tissue}.txt"

        self._run_cmd(cmd)

    @time_decorator(print_args=True)
    def _superenhancers(self, bed: str) -> None:
        """Simple parser to remove superenhancer bed unneeded info"""
        cmd = f" tail -n +2 {self.tissue_dir}/unprocessed/{bed} \
            | awk -v FS='\t' -v OFS='\t' '{{print $1, $2, $3, \"superenhancer\"}}' \
            > {self.tissue_dir}/local/superenhancers_{self.tissue}.bed"

        self._run_cmd(cmd)

    @time_decorator(print_args=True)
    def _tf_binding_sites(self, bed: str) -> None:
        """
        Parse tissue-specific transcription factor binding sites Vierstra et
        al., Nature, 2020, or from Funk et al., Cell Reports, 2020. Funk et al.,
        footprints are 20-seed HINT TFs with score > 200 and use the locations
        of the motifs, not the footprints, as HINT footprints are motif
        agnostic. Motifs are merged to form tf-binding clusters (within 46bp,
        from Chen et al., Scientific Reports, 2015). Vierstra et al., footprints
        are FPR thresholded with p value < 0.001. Footprints within 46bp are
        merged to form clusters, and then intersected with collapsed motifs. If
        90% of the collapsed motif falls within the cluster, the cluster is
        annotated with the binding motif.

        ** Removed clustering of Funk binding sites. The commented code is
        deprecated. **
        """
        # if study == "Funk":
        #     cmd = f"awk -v FS='\t' -v OFS='\t' '$5 >= 200' {self.tissue_dir}/unprocessed/{bed} \
        #         | cut -f1,2,3,4 \
        #         | sed 's/-/\t/g' \
        #         | cut -f1,2,3,6 \
        #         | sed 's/\..*$//g' \
        #         | sort -k1,1 -k2,2n \
        #         | bedtools merge -i - -d 46 -c 4 -o distinct \
        #         > {self.tissue_dir}/local/tfbindingclusters_{self.tissue}.bed"
        # else:
        # cmd = f"cut -f1,2,3 {self.tissue_dir}/unprocessed/{bed} \
        #     | bedtools merge -i - -d 46 \
        #     | bedtools intersect -wa -wb -a - -b {self.resources['tf_motifs']} -F 0.9 \
        #     | bedtools groupby -i - -g 1,2,3 -c 7 -o distinct \
        #     > {self.tissue_dir}/local/tfbindingclusters_{self.tissue}.bed"

        # cmd = f"cut -f1,2,3 {self.tissue_dir}/unprocessed/{bed} \
        #     | bedtools intersect -wa -wb -a - -b {self.resources['tf_motifs']} -F 0.9 \
        #     | bedtools groupby -i - -g 1,2,3 -c 7 -o distinct \
        #     > {self.tissue_dir}/local/tfbindingsites_{self.tissue}.bed"

        # cmd = f"cut -f1,2,3 {self.tissue_dir}/unprocessed/{bed} \
        #     | bedtools intersect -wa -wb -a - -b {self.resources['tf_motifs']} \
        #     | bedtools groupby -i - -g 1,2,3 -c 7 -o distinct \
        #     > {self.tissue_dir}/local/tfbindingsites_{self.tissue}.bed"

        cmd = f"awk -v FS='\t' -v OFS='\t' '{{print $1, $2, $3, \"footprint\"}}' {self.tissue_dir}/unprocessed/{bed} \
            > {self.tissue_dir}/local/tfbindingsites_{self.tissue}.bed"
        rename = f"awk -v FS='\t' -v OFS='\t' '{{print $1, $2, $3, $1\"_\"$2\"_footprint\"}}' {self.tissue_dir}/unprocessed/{bed} \
            > {self.tissue_dir}/unprocessed/tfbindingsites_ref.bed"
        for cmds in [cmd, rename]:
            self._run_cmd(cmds)

    def _liftover(
        self, liftover: str, bed: str, liftover_chain: str, path: str
    ) -> None:
        """Liftovers bed file, sorts it, and deletes the unlifted regions. The
        command is a subprocess call and is set up like:
        
        ./liftOver \
            input.bed \
            hg19ToHg38.over.chain \
            output.bed \
            unlifted.bed
            
        bedtools sort -i output.bed > output_sorted.bed && \
            mv output_sorted output.bed \
            && rm unlifted.bed
            
        Returns none, but creates an output file *path/bed_lifted*
        """
        cmd = f"{liftover} \
            {path}/{bed} \
            {liftover_chain} \
            {path}/{bed}_lifted \
            {path}/{bed}_unlifted \
            && bedtools sort -i {path}/{bed}_lifted \
            > {path}/{bed}_lifted_sorted \
            && mv {path}/{bed}_lifted_sorted {path}/{bed}_lifted \
            && rm {path}/{bed}_unlifted"

        self._run_cmd(cmd)

    @time_decorator(print_args=True)
    def _normalize_mirna(self, file: str) -> None:
        """CPM (counts per million) normalization for miRNA-seq counts"""
        # get miRNA reference
        mirnaref = _mirna_ref(self.references["mirna"])

        mirna = pd.read_csv(
            file,
            sep="\t",
            header=None,
            skiprows=[0, 1, 2, 3],
            usecols=[0, 1],
            names=["gene", "count"],
        )

        # CPM normalization
        mirna = self._count_per_million(mirna)

        # Filter out miRNAs that are not in the reference and flatten list
        active_mirna_gencode = (
            mirna["gene"]
            .apply(lambda x: mirnaref[x] if x in mirnaref and mirnaref[x] else [])
            .tolist()
        )
        flattned_mirna = [item for sublist in active_mirna_gencode for item in sublist]

        # Write out to file
        with open(
            f"{self.tissue_dir}/interaction/active_mirna_{self.tissue}.txt", "w"
        ) as f:
            for item in flattned_mirna:
                f.write("%s\n" % item)

    @time_decorator(print_args=True)
    def _combined_cpg(self, beds: List[str], path: str) -> None:
        """Combined methylation signal across CpGs and average by dividing by
        the amount of files combined."""
        cmd = f"cat {' '.join([f'{path}/{bed}' for bed in beds])} \
                | sort -k1,1, -k2,2n \
                | bedtools merge -i - -c 11 -o mean \
                > {path}/merged_cpgs.bed"

        self._run_cmd(cmd)

    @time_decorator(print_args=True)
    def _bigwig_to_filtered_bedgraph(
        self, path: str, file: str, resource_dir: str
    ) -> str:
        """Convert bigwig to bedgraph file"""
        bed = f"{path}/{file}"
        convert_cmd = f"{resource_dir}/bigWigToBedGraph {bed}.bigwig {bed}.bedGraph"
        filter_cmd = f"awk '$4 >= 0.8' {bed}.bedGraph > {bed}_gt80.bedGraph"
        for cmd in [convert_cmd, filter_cmd]:
            self._run_cmd(cmd)
        return f"{path}/{file}_gt80.bedGraph"

    @time_decorator(print_args=True)
    def _merge_cpg(self, bed: Union[str, List[str]]) -> None:
        """Process CPGs with optional liftover. Bookended methylated CpGs are
        merged. If the config includes multiple CpGs, then methylation signal is
        combined across CpGs and averaged by dividing by the amount of files
        combined."""
        if isinstance(bed, list):
            self._combined_cpg(beds=bed, path=f"{self.tissue_dir}/unprocessed")
            cpg_bed = "merged_cpgs"
            cpg_percent_col = 4
        else:
            cpg_bed = bed
            cpg_percent_col = 11

        if self.methylation["cpg_filetype"] == "roadmap":
            cpg_bed = self._bigwig_to_filtered_bedgraph(
                path=f"{self.tissue_dir}/unprocessed",
                file=cpg_bed.split(".bigwig")[0],
                resource_dir=self.resources["roadmap"],
            )

        if self.methylation["cpg_liftover"] == True:
            self._liftover(
                liftover=self.resources["liftover"],
                bed=cpg_bed,
                liftover_chain=self.resources["liftover_chain"],
                path=f"{self.tissue_dir}/unprocessed",
            )

        if self.methylation["cpg_filetype"] == "ENCODE":
            file = f"{self.tissue_dir}/unprocessed/{bed}_gt80"
            gt_gc = f"awk -v FS='\t' -v OFS='\t' '${cpg_percent_col} >= 80' {self.tissue_dir}/unprocessed/{bed} \
                > {file}"
            self._run_cmd(gt_gc)
        elif self.methylation["cpg_filetype"] == "GEO":
            file = f"{self.tissue_dir}/unprocessed/{bed}_gt80"
            gt_gc = f"sed -e 's/\//\t/g' \
                | tr -d '\ \
                | awk '{{print $4/$5}}' \
                | awk '$4 >= 0.8' \
                > {file}"
        else:
            file = f"{self.tissue_dir}/unprocessed/{bed}"

        bedtools_cmd = f"bedtools merge \
            -i {file} \
            | awk -v FS='\t' -v OFS='\t' '{{print $1, $2, $3, \"cpg_methyl\"}}' \
            > {self.tissue_dir}/local/cpg_{self.tissue}_parsed.bed"

        self._run_cmd(bedtools_cmd)

    @time_decorator(print_args=True)
    def prepare_data_files(self) -> None:
        """Pipeline to prepare all bedfiles.

        Args:
            self: Instance of the class.

        Returns:
            None
        """

        ### Make symlinks for shared data files
        for file in self.shared.values():
            src = f"{self.shared_data_dir}/local/{file}"
            dst = f"{self.tissue_dir}/local/{file}"
            if (
                file in NODETYPES_LOCAL
                and file in self.nodes
                or file not in NODETYPES_LOCAL
            ):
                check_and_symlink(
                    src=src,
                    dst=dst,
                )

        ### Make symlinks for histone marks
        for datatype in self.features:
            check_and_symlink(
                src=f"{self.data_dir}/{self.features[datatype]}",
                dst=f"{self.tissue_dir}/local/{datatype}_{self.tissue}.bed",
            )

        ### Make symlinks for regulatory data
        regulatory_elements = REGULATORY_ELEMENTS[self.regulatory]
        for element in regulatory_elements:
            check_and_symlink(
                src=f"{self.reg_dir}/{regulatory_elements[element]}",
                dst=f"{self.tissue_dir}/local/{element}_{self.tissue}.bed",
            )

        ### Make symlink for cpg
        src = f"{self.data_dir}/{self.methylation['cpg']}"
        dst = f"{self.tissue_dir}/unprocessed/{self.methylation['cpg']}"
        check_and_symlink(
            src=src,
            dst=dst,
        )

        if self.nodes is not None:
            if "crms" in self.nodes:
                check_and_symlink(
                    src=f"{self.data_dir}/{self.tissue_specific_nodes['crms']}",
                    dst=f"{self.tissue_dir}/local/crms_{self.tissue}.bed",
                )
            if "tads" in self.nodes:
                self._add_tad_id(self.tissue_specific_nodes["tads"])
            if "superenhancers" in self.nodes:
                self._superenhancers(self.tissue_specific_nodes["super_enhancer"])
            if "tfbindingsites" in self.nodes:
                self._tf_binding_sites(self.tissue_specific_nodes["tf_binding"])

        self._merge_cpg(self.methylation["cpg"])

        # parse active miRNAs from raw data
        if "mirna" in self.nodes:
            self._normalize_mirna(
                f"{self.tissue_dir}/unprocessed/{self.interaction['mirna']}"
            )

    @staticmethod
    def _count_per_million(df: pd.DataFrame) -> pd.DataFrame:
        """CPM (counts per million) normalization for miRNA-seq counts. Assumes
        there is a column named "count" in the dataframe with gene
        quantifications"""
        try:
            total = df["count"].sum()
        except KeyError as e:
            raise ValueError("DataFrame must have a column named 'count'") from e
        df["cpm"] = (df["count"] / total) * 1e6
        return df[df["cpm"] >= 3]