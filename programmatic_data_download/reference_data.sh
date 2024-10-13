#!/bin/bash
#
# Code to download and parse reference base data for the project. This script
# will assume that your environment has working BEDOPS, BEDTools, and xlsx2csv
# along with Bash > 4.0. Additionally, a python environment with pybedtools
# is required and its PATH should be accessible from wherever you run this
# script. Initial data will be downloaded to the unprocessed directory and
# parsed data placement is hardcoded based on the provided root directory. To
# run the script:
#
# $ bash reference_data.sh \
# $   --root_directory /path/to/project/root \
# $   --postar3_file /path/to/postar3/file \
# $   --script_directory /path/to/programmatic_data_download \
# $   [--cleanup]
# 
# The script will also prepare the necessary directory structure for the rest of
# OGL. Provided is a schematic of the directory layout:
#
# root_directory
# ├── unprocessed
# ├── raw_tissue_data
# │   └── chromatin_loops
# │   └── epimap_tracks
# ├── shared_data
#     ├── local
#     ├── processed_loops
#     ├── regulatory_elements
#     |   └── unprocessed
#     ├── references
#     ├── interaction
#     └── targets
#         ├── expression
#         ├── matrices
#         └── tpm
#
# There is no direct download link for the postar3 data, so users will be
# required to download it manually.


# =============================================================================
# Set up command line variables
# Arguments:
#   --root_directory: project root directory
#   --cleanup: boolean flag to remove intermediate files
#   --postar3_file: path to postar3 file
#   --script_directory: path to programmatic_data_download
# =============================================================================
# Initialize the variables
root_directory=""
postar3_file=""
script_directory=""
cleanup=false

# Parse the command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --root_directory)
            root_directory="$2"
            shift 2
            ;;
        --postar3_file)
            postar3_file="$2"
            shift 2
            ;;
        --script_directory)
            script_directory="$2"
            shift 2
            ;;
        --cleanup)
            cleanup=true
            shift
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done


# =============================================================================
# Setting up variables to track time and progress
# =============================================================================
SECONDS=0

function convertsecs() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(( (total_seconds % 3600) / 60 ))
    local seconds=$((total_seconds % 60))
    printf "%02d:%02d:%02d\n" $hours $minutes $seconds
}


# Function to echo script progress to stdout
log_progress() {
    echo -e "[$(date +%Y-%m-%dT%H:%M:%S%z)] $1\n"
}


# =============================================================================
# Utility function to download raw files if they do not exist
# =============================================================================
function _download_raw_file () {
    local raw_file=$1  # absolute path to raw file
    local download_url=$2  # URL to download raw file

    if [ ! -f ${raw_file} ]; then
        echo "Downloading ${raw_file}..."
        wget -nv -O ${raw_file} ${download_url}
    else
        echo "${raw_file} exists."
    fi
}


# =============================================================================
# Prepare directory folders and subfolders as necessary
# =============================================================================
function _prepare_directory_structure () {
    local root_dir=$1  # project root directory
    local directories=(
        "unprocessed"
        "raw_tissue_data/chromatin_loops"
        "raw_tissue_data/epimap_tracks"
        "shared_data/interaction"
        "shared_data/processed_loops"
        "shared_data/local"
        "shared_data/references"
        "shared_data/regulatory_elements/unprocessed"
        "shared_data/targets/expression"
        "shared_data/targets/tpm"
        "shared_data/targets/matrices"
    )

    mkdir -p ${root_dir}
    for directory in "${directories[@]}"; do
        mkdir -p ${root_dir}/${directory}
    done
}


# =============================================================================
# Convert gencode v26 GTF to bed, remove micro RNA genes and only keep canonical
# "gene" entries. Additionally, keep a lookup table for the conversion from
# gencode ENSG IDs to genesymbol.
# =============================================================================
function _filter_gencode_annotations () {
    local raw_gencode_file=$1  # absolute path to gencode file
    local filtered_gencode_file=$2  # absolute path to gencode bed file

    _download_raw_file \
        ${raw_gencode_file} \
        https://storage.googleapis.com/adult-gtex/references/v8/reference-tables/gencode.v26.GRCh38.genes.gtf

    gtf2bed < ${raw_gencode_file} \
        | awk '$8 == "gene"' \
        | grep -v miR \
        > ${filtered_gencode_file}
}

function _prepare_gencode_lookup_table () {
    local filtered_gencode_file=$1  # absolute path to filtered gencode bed file
    local lookup_table=$2  # absolute path to lookup table

    cut -f4,10 ${filtered_gencode_file} \
        | sed 's/;/\t/g' \
        | cut -f1,5 \
        | sed -e 's/ gene_name //g' -e 's/\"//g' \
        > ${lookup_table}
}


# =============================================================================
# Transcription start sites for hg38 are downloaded from refTSS v4.1, release
# 01/31/2024. First, we create an associate array to store the tss --> gene
# relationships and then we use them to parse a tss file with each gene symbol
# annotated. Lastly, we convert the gene symbols to gencode IDs and only keep
# genes w/ gencode IDs.
# =============================================================================
function _tss () {
    local script_dir=$1  # absolute path to script directory
    local annotation_tss_file=$2  # absolute path to raw tss file
    local raw_tss_file=$3  # absolute path to decompressed tss file
    local decompressed_annotation_file=$4  # absolute path to decompressed tss file
    local decompressed_tss_file=$5  # absolute path to decompressed tss file
    local gencode_ref=$6  # absolute path to gencode reference file
    local parsed_tss_file=$7

    _download_raw_file \
        ${annotation_tss_file} \
        https://reftss.riken.jp/datafiles/4.1/human/refTSS_v4.1_human_hg38_annotation.txt.gz

    _download_raw_file \
        ${raw_tss_file} \
        https://reftss.riken.jp/datafiles/4.1/human/refTSS_v4.1_human_coordinate.hg38.bed.txt.gz

    gunzip -c ${annotation_tss_file} > ${decompressed_annotation_file}
    gunzip -c ${raw_tss_file} > tmp && cat tmp | tail -n +2 > ${decompressed_tss_file} && rm tmp

    # run python script
    python -u ${script_dir}/reftss_parser.py \
        --tss_file ${decompressed_tss_file} \
        --annotation_file ${decompressed_annotation_file} \
        --gencode_ref ${gencode_ref} \
        --outfile ${parsed_tss_file}
}


# =============================================================================
# MicroRNA targets
# We download the entire miRNA catalogue from miRTarBase (version 9.0) and
# filter the miRNAs to only keep homo sapiens relevant miRNAs with functional
# evidence (remove Non-functional MTI).
# =============================================================================
function _mirtarbase_targets () {
    local unprocessed_dir=$1  
    local raw_mirtarbase_file=$2  # absolute path to mirtarbase xlsx
    local raw_mirtarbase_csv=$3  # absolute path to mirtarbase csv
    local parsed_mirtarbase=$4  # absolute path to final parsed file

    if [ ! -f ${raw_mirtarbase_file} ]; then
        echo "Downloading ${raw_mirtarbase_file}..."
        wget -nv -P ${unprocessed_dir} https://mirtarbase.cuhk.edu.cn/~miRTarBase/miRTarBase_2022/cache/download/9.0/miRTarBase_MTI.xlsx
    else
        echo "${raw_mirtarbase_file} exists."
    fi

    # convert to csv
    xlsx2csv ${raw_mirtarbase_file} ${raw_mirtarbase_csv}

    {
        echo -e "miRNA\ttarget_gene"
        cut -f2,4,6,8 -d',' ${raw_mirtarbase_csv} \
            | sed 's/,/\t/g' \
            | awk -vFS='\t' '$3 == "Homo sapiens" && $4 ~ /Functional/' \
            | cut -f1,2 \
            | sort -u
    } > ${parsed_mirtarbase}
}


# =============================================================================
# MicroRNA coordinates 
# We download coordinates of human miRNAs from miRBase release 22.1 For the
# miRBase catalog, we remove unceccessary information and only keep the
# coordinates and name (casefolded). Because there are some repeat entrys
# (primary transcript vs gene body), we collapse any redundant mirnas by keeping
# the larger coordinates. Of not, 66 miRNAs had multiple annotations on
# disparate chromosomes. We removed these 66 miRNA from our analysis. To convert
# between miRNA IDs and ENSG identifiers, we create a lookup table with
# coordinates from biomart.
# =============================================================================
function _mirbase_mirnas () {
    local raw_mirbase_file=$1  # absolute path to raw mirbase file
    local parsed_mirbase_file=$2  # absolute path to parsed mirbase file

    _download_raw_file \
        ${raw_mirbase_file} \
        "https://www.mirbase.org/download/hsa.gff3"

    grep -v "^#" ${raw_mirbase_file} \
        | cut -f1,4,5,9 \
        | awk 'BEGIN{FS=OFS="\t"} {split($4, a, ";"); for (i in a) if (a[i] ~ /^Name=/) {split(a[i], b, "="); $4 = tolower(b[2])}} 1' \
        | sort -k4,4 -k1,1 -k2,2n \
        | bedtools groupby -g 4 -c 1,2,3 -o distinct,min,max \
        | awk 'BEGIN{OFS="\t"} {print $2, $3, $4, $1}' \
        | sort -k1,1 -k2,2n \
        | grep -v "," \
        > ${parsed_mirbase_file}
}


function _biomart_mirna_coordinates () {
    local biomart_file=$1  # absolute path to biomart file
    local mirna_coordinates=$2  # absolute path to mirna_coordinate file

    if [ -f ${biomart_file} ]; then
        echo "Biomart file exists."
    else
        curl -o "${biomart_file}" "https://www.ensembl.org/biomart/martservice" --data-urlencode "query=<?xml version=\"1.0\" encoding=\"UTF-8\"?><!DOCTYPE Query><Query  virtualSchemaName = \"default\" formatter = \"TSV\" header = \"0\" uniqueRows = \"0\" count = \"\" datasetConfigVersion = \"0.6\" ><Dataset name = \"hsapiens_gene_ensembl\" interface = \"default\" ><Attribute name = \"ensembl_gene_id\" /><Attribute name = \"ensembl_gene_id_version\" /><Attribute name = \"chromosome_name\" /><Attribute name = \"start_position\" /><Attribute name = \"end_position\" /><Attribute name = \"external_gene_name\" /><Attribute name = \"mirbase_accession\" /><Attribute name = \"mirbase_id\" /></Dataset></Query>"
    fi

    awk '{ if (length($8) > 0) print $0 }' ${biomart_file} \
        | awk 'BEGIN{FS=OFS="\t"} {print "chr"$3, $4, $5, $2, $8}' \
        | grep -v "PATCH" \
        | awk '$1 != "chrMT" && $1 != "chrY" && $1 != "chrX"' \
        | sort -k1,1 -k2,2n \
        > ${mirna_coordinates}
}


# =============================================================================
# NOTE - there is no direct download for POSTAR 3 sites. Users must download
# ahead of time.
# RNA Binding protein sites were downloaded from POSTAR 3. We take the list of
# RBPs and their binding sites and intersect their binding sites with gene
# bodies. To get RBP --> Gene interactions. We then keep RBP --> Gene
# interactions that occur in multiple samples, removing singletons. 
#
# To determine RBP binding site clusters, we calculate the node feature (overlap
# with rna binding site clusters), we merge binding sites within 10bp, given
# that RBP binding sites are often small yet clustered (see Li et al., Genome
# Biology, 2017). To keep higher confidence sites we apply two filters: (1) we
# only keep sites that target multiple target genes and (2) we ensure that the
# site exists in at least 3 different samples, because filtering at 2 samples
# will still keep singletons (e.g. HEK293T,HEK_293_FRT could effectively be
# considered the same sample).
# =============================================================================
function _rbp_binding_sites () {
    local raw_postar3_gunzipped=$1  # absolute path to raw postar3 file
    local raw_postar3_txt=$2  # absolute path to gunzipped postar3 file
    local parsed_binding_sites=$3  # absolute path to parsed binding sites
    local gencode=$4  # absolute path to gencode file
    local gencode_lookup=$5  # absolute path to gencode lookup table
    local rbp_network=$6  # absolute path to rbp network in edge list format

    gunzip -c ${raw_postar3_gunzipped} > ${raw_postar3_txt}

    awk '$6 != "RBP_occupancy"' ${raw_postar3_txt} \
        | cut -f1,2,3,6,8 \
        | bedtools intersect \
        -a - \
        -b ${gencode} \
        -wa \
        -wb \
        | cut -f4,5,9 \
        | sort -u \
        | awk 'BEGIN{FS=OFS="\t"} {array[$1 OFS $3] = array[$1 OFS $3] ? array[$1 OFS $3] "," $2 : $2} END{for (i in array) print i, array[i]}' \
        | grep "," \
        > ${parsed_binding_sites}

    awk 'BEGIN { FS=OFS="\t" } NR==FNR { lookup[$2]=$1; next } $1 in lookup { $1=lookup[$1] } split($3, a, ",") >= 3 { print }' \
        ${gencode_lookup} \
        ${parsed_binding_sites} \
        | grep -e '^ENSG' \
        > ${rbp_network}
}


function _rbp_site_clusters () {
    local raw_postar3_txt=$1  # absolute path to gunzipped postar3 file
    local parsed_binding_sites=$2  # absolute path to parsed binding sites

    awk '$6 != "RBP_occupancy"' ${raw_postar3_txt} \
        | bedtools merge \
        -d 10 \
        -i - \
        -c 6,6,8,8 \
        -o distinct,count_distinct,count_distinct,distinct \
        | awk '($5 > 1) && ($6 > 3)' \
        | cut -f1,2,3,4 \
        | sed 's/,/_/g' \
        | awk -v OFS='\t' '{print $1, $2, $3, "rnab_"$4}' \
        > ${parsed_binding_sites}
}


# =============================================================================
# Make node_attribute references for gencode, superenhancer, and rbp data. These
# references exist as a lookup table for node attributes downdstream during
# graph construction, which allows us to go from an edge list to a list of
# basenodes and their locations.
# The first part of the function adds attrs for gencode nodes but adds duplicate
# entries to differentiate the protein or tf as opposed to the gene body.
# =============================================================================
function _node_featsaver () {
    local gencode_file=$1  # absolute path to gencode file
    local genocode_attr=$2  # absolute path to gencode node attribute file
    local sedb_list=$3  # absolute path to superenhancer database file
    local sedb_attr=$4  # absolute path to superenhancer node attribute file

    # <(awk -v OFS='\t' '{print $1, $2, $3, $4"_protein"}' $1/$2) \
    cat <(awk -v OFS='\t' '{print $1, $2, $3, $4}' ${gencode_file}) \
        <(awk -v OFS='\t' '{print $1, $2, $3, $4"_tf"}' ${gencode_file}) \
        > ${genocode_attr}

    _download_raw_file \
        ${sedb_list} \
        https://bio.liclab.net/sedb/download/new/package/SE_package_hg38.bed

    tail -n +2 ${sedb_list} \
        | awk -v OFS='\t' '{print $3,$4,$5,$3"_"$4"_superenhancer"}' \
        > ${sedb_attr}
}


# =============================================================================
# Main function! Takes care of all the data processing and downloading.
# =============================================================================
function main () {
    log_progress "Setting up input vars"
    local root_dir=$1  # project root directory
    local postar3_file=$2  # path to postar3 file
    local script_dir=$3  # script directory

    # Set up directory paths
    unprocessed_dir="$root_dir/unprocessed"
    interaction_dir="$root_dir/shared_data/interaction"
    local_dir="$root_dir/shared_data/local"
    reference_dir="$root_dir/shared_data/references"
    regulatory_elements_dir="$root_dir/shared_data/regulatory_elements"
    expression_dir="$root_dir/shared_data/targets/expression"
    tpm_dir="$root_dir/shared_data/targets/tpm"
    matrices_dir="$root_dir/shared_data/targets/matrices"

    log_progress "Preparing directory structure"
    _prepare_directory_structure \
        "$root_dir"

    log_progress "Download files that do not require processing"
    declare -A files_to_download=(
        ["$reference_dir/hg38.chrom.sizes"]="https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/hg38.chrom.sizes"
        ["$reference_dir/hg19.chrom.sizes"]="https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.chrom.sizes"
        ["$reference_dir/liftOver"]="http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/liftOver"
        ["$reference_dir/bigWigToWig"]="http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/bigWigToWig"
        ["$reference_dir/bigWigToBedGraph"]="http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/bigWigToBedGraph"
        ["$reference_dir/bigBedToBed"]="http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/bigBedToBed"
        ["$reference_dir/peakMerge.py"]="https://github.com/remap-cisreg/peakMerge/raw/main/peakMerge.py"
        ["$reference_dir/hg19ToHg38.over.chain.gz"]="https://hgdownload.cse.ucsc.edu/goldenpath/hg19/liftOver/hg19ToHg38.over.chain.gz"
        ["$reference_dir/yue_hg38_tads.zip"]="http://3dgenome.fsm.northwestern.edu/downloads/hg38.TADs.zip"
        ["$reference_dir/peakachu_loops.zip"]="http://3dgenome.fsm.northwestern.edu/downloads/loops-hg38.zip"
        ["$unprocessed_dir/hg38.fa.gz"]="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
        ["$unprocessed_dir/hg38-blacklist.v2.bed.gz"]="https://github.com/Boyle-Lab/Blacklist/raw/master/lists/hg38-blacklist.v2.bed.gz"
        ["$unprocessed_dir/collapsed_motifs_overlapping_consensus_footprints_hg38.bed.gz"]="https://resources.altius.org/~jvierstra/projects/footprinting.2020/consensus.index/collapsed_motifs_overlapping_consensus_footprints_hg38.bed.gz"
    )

    # download each file
    for target_path in "${!files_to_download[@]}"; do
        _download_raw_file "$target_path" "${files_to_download[$target_path]}"
    done

    # set execution permission for UCSC utilities
    chmod +x "$reference_dir/liftOver"
    chmod +x "$reference_dir/bigWigToWig"
    chmod +x "$reference_dir/bigWigToBedGraph"
    chmod +x "$reference_dir/bigBedToBed"

    log_progress "Decompressing zip files"
    gunzip -c "$unprocessed_dir/hg38.fa.gz" > "$reference_dir/hg38.fa"
    gunzip -c "$unprocessed_dir/hg38-blacklist.v2.bed.gz" > "$reference_dir/hg38-blacklist.v2.bed"
    gunzip -c "$unprocessed_dir/collapsed_motifs_overlapping_consensus_footprints_hg38.bed.gz" > "$reference_dir/collapsed_motifs_overlapping_consensus_footprints_hg38.bed"

    unzip -o -d "$reference_dir" "$reference_dir/yue_hg38_tads.zip"
    mv "$reference_dir/hg38" "$reference_dir/yue_hg38_tads"

    unzip -o -d "$reference_dir" "$reference_dir/peakachu_loops.zip"
    mv "$reference_dir/hg38" "$reference_dir/peakachu_loops"

    log_progress "Prepare gencode related files"
    _filter_gencode_annotations \
        "$unprocessed_dir/gencode.v26.GRCh38.genes.gtf" \
        "$reference_dir/gencode_v26_genes_only_with_GTEx_targets.bed"

    _prepare_gencode_lookup_table \
        "$reference_dir/gencode_v26_genes_only_with_GTEx_targets.bed" \
        "$reference_dir/gencode_to_genesymbol_lookup_table.txt"

    # make a symlink and place gencode file in local
    ln -s \
        "$reference_dir/gencode_v26_genes_only_with_GTEx_targets.bed" \
        "$local_dir/gencode_v26_genes_only_with_GTEx_targets.bed"

    log_progress "Prepare TSS files"
    _tss \
        "$script_dir" \
        "$unprocessed_dir/refTSS_v4.1_human_hg38_annotation.txt.gz" \
        "$unprocessed_dir/refTSS_v4.1_human_coordinate.hg38.bed.txt.gz" \
        "$unprocessed_dir/refTSS_v4.1_human_hg38_annotation.txt" \
        "$unprocessed_dir/refTSS_v4.1_human_coordinate.hg38.bed.txt" \
        "$reference_dir/gencode_v26_genes_only_with_GTEx_targets.bed" \
        "$local_dir/tss_parsed_hg38.bed"

    log_progress "Prepare miRTarBase files"
    _mirtarbase_targets \
        "$unprocessed_dir" \
        "$unprocessed_dir/miRTarBase_MTI.xlsx" \
        "$unprocessed_dir/miRTarBase_MTI.csv" \
        "$reference_dir/mirtargets_filtered.txt"

    log_progress "Prepare miRBase miRNA files"
    _mirbase_mirnas \
        "$unprocessed_dir/hsa.gff3" \
        "$reference_dir/mirbase_mirnas.bed"

    _biomart_mirna_coordinates \
        "$unprocessed_dir/mart_export.txt" \
        "$reference_dir/ensembl_mirna_coordinates_hg38.bed"

    log_progress "Prepare RBP network"
    _rbp_binding_sites \
        "$postar3_file" \
        "$unprocessed_dir/human.txt" \
        "$unprocessed_dir/rbp_gene_binding_sites.bed" \
        "$reference_dir/gencode_v26_genes_only_with_GTEx_targets.bed" \
        "$reference_dir/gencode_to_genesymbol_lookup_table.txt" \
        "$reference_dir/rbp_gene_network.txt"

    _rbp_site_clusters \
        "$unprocessed_dir/human.txt" \
        "$local_dir/rbpsiteclusters_parsed_hg38.bed"

    log_progress "Prepare node attribute files"
    _node_featsaver \
        "$reference_dir/gencode_v26_genes_only_with_GTEx_targets.bed" \
        "$reference_dir/gencode_v26_node_attr.bed" \
        "$unprocessed_dir/SE_package_hg38.bed" \
        "$reference_dir/se_node_attr.bed"

    log_progress "Downloading TF marker, remove unecessary columns, "
    wget -O \
        ${unprocessed_dir}/tf_marker.txt \
        https://bio.liclab.net/TF-Marker/public/download/main_download.csv

    grep "Normal cell" ${unprocessed_dir}/tf_marker.txt \
        | cut -f1,2,3,4,5,11 \
        > ${reference_dir}/tf_marker.bed
}


# =============================================================================
# Run the script, given that arguments are passed properly.
# =============================================================================
# check if the root_directory is not set
if [[ -z "$root_directory" ]] || [[ -z "$postar3_file" ]] || [[ -z "$script_directory" ]]; then
    echo "Error: --root_directory and/or --postar3_file and/or --script_directory are not set."
    echo "Usage: reference_data.sh --root_directory PATH --postar3_file PATH [--cleanup]"
    exit 1
else
    echo "Root directory is set to $root_directory"
    echo "Postar3 file is set to $postar3_file"
    echo "Script directory is set to $script_directory"
    main "${root_directory}" "${postar3_file}" "${script_directory}"
fi

# run optional cleanup
if [[ "$cleanup" == true ]]; then
    echo "Cleanup boolean is set to true. Removing intermediate files..."
    rm -r "$root_directory/unprocessed"
fi

echo "Finished preparing reference data. Total time: $(convertsecs SECONDS)"


# NOTE some of the tads have malformed lines, where the end is greater than the start.
# you can inspect it via:
    # for file in *; do echo $file; awk '{ if ($2 > $3) print NR ": " $0 }' $file; done
#
# we removed the lines with:
# for file in *; do
#     echo "Processing $file"
    
#     # check if there are any malformed lines
#     if awk '{ if ($2 > $3) exit 1 }' "$file"; then
#         echo "No malformed lines found in $file"
#     else
#         echo "Malformed lines found in $file. Cleaning..."
#         # remove malformed lines in-place
#         awk '{ if ($2 <= $3) print $0 }' "$file" > temp_file && mv temp_file "$file"
#     fi
# done
