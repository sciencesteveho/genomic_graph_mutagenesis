#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Parse local genomic data to generate nodes and attributes. Given basenodes
from edge parsing, this module will generate edges based on the defined context
window (i.e. 5kb). Additionally, node features are processed here as they also
derive from the local context datatypes."""


from itertools import repeat
from multiprocessing import Pool
import os
from pathlib import Path
import pickle
import subprocess
from subprocess import PIPE
from subprocess import Popen
from typing import Any, Dict, List, Optional, Tuple, Union

from pybedtools import BedTool  # type: ignore
from pybedtools.featurefuncs import extend_fields  # type: ignore

from omics_graph_learning.positional_encoding import PositionalEncoding
from omics_graph_learning.utils.common import _chk_file_and_run
from omics_graph_learning.utils.common import dir_check_make
from omics_graph_learning.utils.common import genes_from_gencode
from omics_graph_learning.utils.common import get_physical_cores
from omics_graph_learning.utils.common import setup_logging
from omics_graph_learning.utils.common import time_decorator
from omics_graph_learning.utils.config_handlers import ExperimentConfig
from omics_graph_learning.utils.config_handlers import TissueConfig
from omics_graph_learning.utils.constants import ATTRIBUTES

logger = setup_logging()


class LocalContextParser:
    """Object that parses local genomic data into graph edges, connected all
    nodes within a given linear context window.

    Attributes:
        experiment_config: Dataclass containing the experiment configuration.
        tissue_config: Dataclass containing the tissue configuration.
        bedfiles: dictionary containing each local genomic data file

    Methods
    ----------
    _prepare_local_features:
        Creates a dict of local context datatypes and their bedtools objects.
    _process_bedcollection:
        Process a collection of bedfiles by sorting then adding a window (slop)
        around each feature.
    _bed_intersect:
        Function to intersect a slopped bed entry with all other node types.
    _aggregate_attributes:
        For each node of a node_type get their overlap with gene windows then
        aggregate total nucleotides, gc content, and all other attributes.
    _generate_edges:
        Unix concatenate and sort each edge file.
    _save_node_attributes:
        Save attributes for all node entries.
    parse_context_data:
        Parse local genomic data into graph edges.

    # Helpers
        ATTRIBUTES -- list of node attribute types
        DIRECT -- list of datatypes that only get direct overlaps, no slop
        ALL_CONCATENATED_FILE -- name of concatenated file
        SORTED_CONCATENATED_FILE -- name of sorted concatenated file

    Examples:
    --------
    >>> localparseObject = LinearContextParser(
        experiment_config=experiment_config,
        tissue_config=tissue_config,
        bedfiles=adjusted_bedfiles,
    )

    >>> localparseObject.parse_context_data()
    """

    # var helpers
    all_concatenated_file = "all_concat.bed"
    sorted_concatenated_file = "all_concat_sorted.bed"

    # list helpers
    direct = ["tads", "loops"]
    node_feats = ["start", "end", "size"] + ATTRIBUTES

    # helper for CPU cores
    cores = get_physical_cores()

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        tissue_config: TissueConfig,
        bedfiles: List[str],
    ):
        """Initialize the class"""
        self.bedfiles = bedfiles
        self.blacklist = experiment_config.blacklist
        self.chromfile = experiment_config.chromfile
        self.experiment_name = experiment_config.experiment_name
        self.fasta = experiment_config.fasta
        self.feat_window = experiment_config.feat_window
        self.nodes = experiment_config.nodes
        self.working_directory = experiment_config.working_directory
        self.build_positional_encoding = experiment_config.build_positional_encoding

        self.resources = tissue_config.resources
        self.gencode = tissue_config.local["gencode"]
        self.tissue = tissue_config.resources["tissue"]
        self.tss = tissue_config.local["tss"]

        self.node_processes = len(self.nodes) + 1
        self._set_directories()
        self._prepare_bed_references()

        # make directories
        self._make_directories()

    def _set_directories(self) -> None:
        """Set directories from yaml"""
        self.tissue_dir = self.working_directory / self.tissue
        self.local_dir = self.tissue_dir / "local"
        self.parse_dir = self.tissue_dir / "parsing"
        self.attribute_dir = self.parse_dir / "attributes"

        self.edge_dir = self.parse_dir / "edges"
        self.intermediate_dir = self.parse_dir / "intermediate"
        self.intermediate_sorted = self.intermediate_dir / "sorted"

    def _prepare_bed_references(
        self,
    ) -> None:
        """Prepares genesymbol to gencode conversion and blacklist as pybedtools
        objects.
        """
        self.genesymbol_to_gencode = genes_from_gencode(
            BedTool(self.local_dir / self.gencode)  # type: ignore
        )
        self.blacklist = BedTool(self.blacklist).sort().saveas()  # type: ignore

    def _make_directories(self) -> None:
        """Directories for parsing genomic bedfiles into graph edges and nodes"""
        dir_check_make(self.parse_dir)

        for directory in [
            "edges",
            "attributes",
            "intermediate/slopped",
            "intermediate/sorted",
        ]:
            dir_check_make(self.parse_dir / directory)

        for attribute in ATTRIBUTES:
            dir_check_make(self.attribute_dir / attribute)

    def _initialize_positional_encoder(
        self,
    ) -> PositionalEncoding:
        """Get object to produce positional encodings."""
        return PositionalEncoding(
            chromfile=self.chromfile,
            binsize=50000,
        )

    @time_decorator(print_args=True)
    def parse_context_data(self) -> None:
        """Parse local genomic data into graph edges."""
        # process windows and renaming
        with Pool(processes=self.node_processes) as pool:
            bedcollection_flat = dict(
                pool.starmap(
                    self._prepare_local_features, [(bed,) for bed in self.bedfiles]
                )
            )

        # sort and extend windows according to FEAT_WINDOWS
        bedcollection_sorted, bedcollection_slopped = self.process_bedcollection(
            bedcollection=bedcollection_flat,
            chromfile=self.chromfile,
            feat_window=self.feat_window,
        )

        # save intermediate files
        self._save_intermediate(bedcollection_sorted, folder="sorted")
        self._save_intermediate(bedcollection_slopped, folder="slopped")

        # pre-concatenate to save time
        all_files = self.intermediate_sorted / "all_files_concatenated.bed"
        self._pre_concatenate_all_files(all_files, bedcollection_slopped)

        # perform intersects across all feature types - one process per nodetype
        with Pool(processes=self.node_processes) as pool:
            pool.starmap(self._bed_intersect, zip(self.nodes, repeat(all_files)))

        # get size and all attributes - one process per nodetype
        with Pool(processes=self.cores) as pool:
            pool.map(self._aggregate_attributes, ["basenodes"] + self.nodes)

        # parse edges into individual files
        self._generate_edges()

        # save node attributes as reference for later - one process per nodetype
        with Pool(processes=self.cores) as pool:
            pool.map(self._save_node_attributes, ["basenodes"] + self.nodes)

        # cleanup
        self._cleanup_edge_files()

    @time_decorator(print_args=True)
    def _prepare_local_features(
        self,
        bed: str,
    ) -> Tuple[str, BedTool]:
        """
        Creates a dict of local context datatypes and their bedtools objects.
        Renames features if necessary. Intersects each bed to get only features
        that do not intersect ENCODE blacklist regions.
        """

        def rename_feat_chr_start(feature: Any) -> str:
            """Add chr, start to feature name
            Cpgislands add prefix to feature names  # enhancers,
            Histones add an additional column
            """
            simple_rename = [
                "cpgislands",
                "crms",
            ]
            if prefix in simple_rename:
                feature = extend_fields(feature, 4)
                feature[3] = f"{feature[0]}_{feature[1]}_{prefix}"
            else:
                # ensure feat does not already have chr_start
                parts = feature[3].split("_")
                if (
                    len(parts) >= 3
                    and parts[0] == feature[0]
                    and parts[1] == feature[1]
                ):
                    return feature
                feature[3] = f"{feature[0]}_{feature[1]}_{feature[3]}"
            return feature

        # prepare data as pybedtools objects and intersect -v against blacklist
        local_bed = BedTool(self.local_dir / bed).sort()
        ab = self._remove_blacklist_and_alt_configs(
            bed=local_bed, blacklist=self.blacklist
        )

        # rename features if necessary and only keep coords
        prefix = bed.split("_")[0].lower()
        if prefix in self.nodes and prefix != "gencode":
            result = ab.each(rename_feat_chr_start).cut([0, 1, 2, 3]).saveas()
            prepared_bed = BedTool(str(result), from_string=True)
        else:
            prepared_bed = ab.cut([0, 1, 2, 3])

        return prefix, prepared_bed

    @time_decorator(print_args=True)
    def process_bedcollection(
        self,
        bedcollection: Dict[str, BedTool],
        chromfile: str,
        feat_window: int,
    ) -> Tuple[Dict[str, BedTool], Dict[str, BedTool]]:
        """Process bedfiles by sorting then adding a window (slop) around each
        feature.
        """
        bedcollection_sorted = self._sort_bedcollection(bedcollection)
        bedcollection_slopped = self._slop_bedcollection(
            bedcollection, chromfile, feat_window
        )
        return bedcollection_sorted, bedcollection_slopped

    def _slop_bedcollection(
        self,
        bedcollection: Dict[str, BedTool],
        chromfile: str,
        feat_window: int,
    ) -> Dict[str, BedTool]:
        """Slop bedfiles that are not 3D chromatin files."""
        return {
            key: self._slop_and_keep_original(value, chromfile, feat_window)
            for key, value in bedcollection.items()
            if key not in ATTRIBUTES + self.direct
        }

    @time_decorator(print_args=True)
    def _bed_intersect(
        self,
        node_type: str,
        all_files: str,
    ) -> None:
        """Slopped nodes are intersected with all other node types as a simple
        and fast way to generate edges. Special instances are 3d chromatin based
        nodes, which require a direct intersect instead of a slop.

        Edge files are deduplicated and saved to the edge directory.
        """
        logger.info(f"starting combinations {node_type}")

        def _unix_intersect(node_type: str, type: Optional[str] = None) -> None:
            """Perform a bed intersect using shell, which can be faster than
            pybedtools for large files."""
            if type == "direct":
                folder = "sorted"
                cut_cmd = ""
            else:
                folder = "slopped"
                cut_cmd = " | cut -f5,6,7,8,9,10,11,12"

            final_cmd = f"bedtools intersect \
                -wa \
                -wb \
                -sorted \
                -a {self.intermediate_dir}/{folder}/{node_type}.bed \
                -b {all_files}"

            with open(self.edge_dir / f"{node_type}.bed", "w") as outfile:
                subprocess.run(final_cmd + cut_cmd, stdout=outfile, shell=True)
            outfile.close()

        def _add_distance(feature: Any) -> str:
            """Add distance as [8]th field to each overlap interval"""
            feature = extend_fields(feature, 9)
            feature[8] = max(int(feature[1]), int(feature[5])) - min(
                int(feature[2]), int(feature[5])
            )
            return feature

        if node_type in self.direct:
            _unix_intersect(node_type, type="direct")
            self._filter_duplicate_bed_entries(
                BedTool(self.edge_dir / f"{node_type}.bed")
            ).sort().saveas(self.edge_dir / f"{node_type}_dupes_removed")
        else:
            _unix_intersect(node_type)
            self._filter_duplicate_bed_entries(
                BedTool(self.edge_dir / f"{node_type}.bed")
            ).each(_add_distance).saveas().sort().saveas(
                self.edge_dir / f"{node_type}_dupes_removed"
            )

    @time_decorator(print_args=True)
    def _aggregate_attributes(self, node_type: str) -> None:
        """For each node of a node_type get their overlap with gene windows then
        aggregate total nucleotides, gc content, and all other attributes."""
        # prepare reference file
        ref_file = self._reference_nodes_for_feature_aggregation(node_type)

        # aggregate
        for attribute in ATTRIBUTES:
            save_file = (
                self.attribute_dir / attribute / f"{node_type}_{attribute}_percentage"
            )
            logger.info(f"Processing {attribute} for {node_type}")
            self._group_attribute(ref_file, attribute, save_file)

    def _reference_nodes_for_feature_aggregation(self, node_type: str) -> BedTool:
        """Prepare node_type reference for aggregating attributes"""

        def add_size(feature: Any) -> str:
            """Add size as a field to the bedtool object"""
            feature = extend_fields(feature, 5)
            feature[4] = feature.end - feature.start
            return feature

        return (
            BedTool(self.intermediate_sorted / f"{node_type}.bed")
            .filter(lambda x: "alt" not in x[0])
            .each(add_size)
            .sort()
            .saveas()
        )

    def _group_attribute(
        self, ref_file: BedTool, attribute: str, save_file: Path
    ) -> None:
        """Get overlap with gene windows then aggregate total nucleotides, gc
        content, and all other attributes.
        """

        def sum_gc(feature: Any) -> str:
            """Sum GC content for each feature"""
            feature[13] = int(feature[8]) + int(feature[9])
            return feature

        if attribute == "gc":
            return (
                ref_file.nucleotide_content(fi=self.fasta)
                .each(sum_gc)
                .sort()
                .groupby(g=[1, 2, 3, 4], c=[5, 14], o=["sum"])
                .saveas(save_file)
            )
        elif attribute == "recombination":
            return (
                ref_file.intersect(
                    BedTool(self.intermediate_sorted / f"{attribute}.bed"),
                    wao=True,
                    sorted=True,
                )
                .groupby(g=[1, 2, 3, 4], c=[5, 9], o=["sum", "mean"])
                .sort()
                .saveas(save_file)
            )
        else:
            return (
                ref_file.intersect(
                    BedTool(self.intermediate_sorted / f"{attribute}.bed"),
                    wao=True,
                    sorted=True,
                )
                .groupby(g=[1, 2, 3, 4], c=[5, 10], o=["sum"])
                .sort()
                .saveas(save_file)
            )

    @time_decorator(print_args=True)
    def _generate_edges(self) -> None:
        """Unix concatenate and sort each edge file with parallelization options."""
        cmds = {
            "cat_cmd": [
                f"cat {self.edge_dir}/*_dupes_removed* >",
                f"{self.edge_dir}/{self.all_concatenated_file}",
            ],
            "sort_cmd": [
                f"LC_ALL=C sort --parallel={self.cores} -S 70% -k10,10 {self.edge_dir}/{self.all_concatenated_file} |",
                "uniq >" f"{self.edge_dir}/{self.sorted_concatenated_file}",
            ],
        }
        for cmd in cmds:
            _chk_file_and_run(
                cmds[cmd][1],
                cmds[cmd][0] + cmds[cmd][1],
            )

    def _save_node_attributes(self, node: str) -> None:
        """Save attributes for all node entries to create node attributes during
        graph construction."""
        # process attributes
        stored_attributes = self._process_node_attributes(node)

        # save
        with open(self.attribute_dir / f"{node}_reference.pkl", "wb") as output:
            pickle.dump(stored_attributes, output)

    def _read_attribute_file(self, filename: Path) -> set:
        """Read the attribute file and return their aggregated contents."""
        with open(filename, "r") as file:
            return {tuple(line.rstrip().split("\t")) for line in file}

    def _process_node_attributes(
        self, node: str
    ) -> Dict[str, Dict[str, Union[str, float, int, Dict[str, Union[str, int]]]]]:
        """Add node attributes to reference dictionary for each feature type."""
        # initialize positional encoder
        positional_encoding = (
            self._initialize_positional_encoder()
            if self.build_positional_encoding
            else None
        )

        stored_attributes = {}
        for attribute in ATTRIBUTES:
            filename = self.attribute_dir / attribute / f"{node}_{attribute}_percentage"
            lines = self._read_attribute_file(filename)
            for line in lines:
                try:
                    node_key = f"{line[3]}_{self.tissue}"
                    if attribute == "gc":
                        stored_attributes[node_key] = self._add_first_attribute(
                            line=line, positional_encoding=positional_encoding
                        )
                    else:
                        self._add_remaining_attributes(
                            stored_attributes=stored_attributes,
                            node_key=node_key,
                            attribute=attribute,
                            line=line,
                        )
                except Exception as e:
                    logger.error(f"Error processing {node} {attribute} {line}: {e}")
                    raise
        return stored_attributes

    def _add_first_attribute(
        self,
        line: Tuple[str, ...],
        positional_encoding: Optional[PositionalEncoding],
    ) -> Dict[str, Union[str, float, int, Dict[str, Union[str, int]]]]:
        """Because gc is the first attribute processed, we take this time to
        initialize some of the dictionary values and produce the positional
        encodings."""
        try:
            attributes: Dict[
                str, Union[str, float, int, Dict[str, Union[str, int]]]
            ] = {
                "coordinates": {
                    "chr": line[0],
                    "start": int(line[1]),
                    "end": int(line[2]),
                },
                "size": int(line[4]),
                "gc": int(line[5]),
            }

            if positional_encoding:
                self._add_positional_encoding_attribute(
                    line=line,
                    positional_encoding=positional_encoding,
                    attributes=attributes,
                )
            return attributes
        except Exception as e:
            logger.error(f"Error processing gc {line}: {e}")
            raise

    def _add_remaining_attributes(
        self,
        stored_attributes: Dict[
            str, Dict[str, Union[str, float, int, Dict[str, Union[str, int]]]]
        ],
        node_key: str,
        attribute: str,
        line: tuple,
    ) -> None:
        """Add remaining attributes to the stored attributes dictionary. If the
        attribute aggregate provided no value, defaults to 0."""
        if node_key not in stored_attributes:
            stored_attributes[node_key] = {}
        try:
            stored_attributes[node_key][attribute] = float(line[5])
        except ValueError:
            stored_attributes[node_key][attribute] = 0

    def _remove_blacklist_and_alt_configs(
        self,
        bed: BedTool,
        blacklist: BedTool,
    ) -> BedTool:
        """Remove blacklist and alternate chromosomes from bedfile."""
        return self._remove_alt_configs(bed.intersect(blacklist, sorted=True, v=True))

    @time_decorator(print_args=True)
    def _save_intermediate(
        self, bed_dictionary: Dict[str, BedTool], folder: str
    ) -> None:
        """Save region specific bedfiles"""
        for key in bed_dictionary:
            file = self.parse_dir / "intermediate" / f"{folder}/{key}.bed"
            if not os.path.exists(file):
                bed_dictionary[key].saveas(file)

    @time_decorator(print_args=True)
    def _pre_concatenate_all_files(
        self, all_files: str, bedcollection_slopped: Dict[str, BedTool]
    ) -> None:
        """Pre-concatenate via unix commands to save time"""
        if not os.path.exists(all_files) or os.stat(all_files).st_size == 0:
            cat_cmd = ["cat"] + [
                self.intermediate_sorted / f"{x}.bed" for x in bedcollection_slopped
            ]
            sort_cmd = "sort -k1,1 -k2,2n"
            concat = Popen(cat_cmd, stdout=PIPE)
            with open(all_files, "w") as outfile:
                subprocess.run(
                    sort_cmd, stdin=concat.stdout, stdout=outfile, shell=True
                )
            outfile.close()

    def _cleanup_edge_files(self) -> None:
        """Remove intermediate files"""
        retain = [self.sorted_concatenated_file]
        for item in os.listdir(self.edge_dir):
            if item not in retain:
                os.remove(self.edge_dir / item)

    @staticmethod
    def _add_positional_encoding_attribute(
        line: Tuple[str, ...],
        positional_encoding: PositionalEncoding,
        attributes: Dict[str, Union[str, float, int, Dict[str, Union[str, int]]]],
    ) -> None:
        """Add positional encoding to the attributes dictionary."""
        chr_val = str(line[0])
        start_val = int(line[1])
        end_val = int(line[2])
        attributes["positional_encoding"] = positional_encoding(
            chromosome=chr_val, node_start=start_val, node_end=end_val
        )

    @staticmethod
    def _sort_bedcollection(bedcollection: Dict[str, BedTool]) -> Dict[str, BedTool]:
        """Sort bedfiles in a collection"""
        return {key: value.sort() for key, value in bedcollection.items()}

    @staticmethod
    def _slop_and_keep_original(
        bedfile: BedTool, chromfile: str, feat_window: int
    ) -> BedTool:
        """Slop a single bedfile and re-write each line to keep the original
        entry."""
        slopped = bedfile.slop(g=chromfile, b=feat_window).sort()
        newstrings = [
            str(line_1).split("\n")[0] + "\t" + str(line_2)
            for line_1, line_2 in zip(slopped, bedfile)
        ]
        return BedTool("".join(newstrings), from_string=True).sort()

    @staticmethod
    def _remove_alt_configs(
        bed: BedTool,
    ) -> BedTool:
        """Remove alternate chromosomes from bedfile."""
        return bed.filter(lambda x: "_" not in x[0]).saveas()

    @staticmethod
    def _filter_duplicate_bed_entries(
        bedfile: BedTool,
    ) -> BedTool:
        """Filters a bedfile by removing entries that are identical"""
        return bedfile.filter(
            lambda x: [x[0], x[1], x[2], x[3]] != [x[4], x[5], x[6], x[7]]
        ).saveas()
