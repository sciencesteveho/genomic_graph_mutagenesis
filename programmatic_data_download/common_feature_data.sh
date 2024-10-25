#!/bin/bash
#
# Code to download and parse catalogues of regulatory elements. We take
# regulatory catalogues from:
# 1 - SCREEN registry of cCREs V3
# 2 - EpiMap Repository
# And use four different combinations: SCREEN only, EpiMap only, an
# intersection, and a union of elements. The EpiMap catalogue is in hg18 and is
# lifted over to hg38.
# 
# $ bash common_feature_data.sh \
# $   --root_directory /path/to/your/root/directory
#
# This script assumes that your environment has working BEDOPS and BEDTOOLS.


# =============================================================================
# Set up command line variables
# Arguments:
#   --root_directory: project root directory
# =============================================================================
# Initialize the variables
root_directory=""
cleanup=false

# Parse the command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --root_directory)
            root_directory="$2"
            shift 2
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
# Utility function to perform reference liftover 
# =============================================================================
function _liftover_19_to_38 () {
    local liftover_dir="$1"  # path/to/liftover/directory
    local unprocessed_dir="$2"  # path/to/unprocessed/directory
    local file_name="$3"  # name of file to liftover

    ${liftover_dir}/liftOver \
        ${unprocessed_dir}/${file_name}.bed \
        ${liftover_dir}/hg19ToHg38.over.chain.gz \
        ${unprocessed_dir}/${file_name}.lifted_hg38.bed \
        ${unprocessed_dir}/${file_name}.unlifted
}


# =============================================================================
# poly-(A) target sites for hg38 are downloaded from PolyASite, homo sapiens
# v2.0, release 21/04/2020
# =============================================================================
function _poly_a () {
    local unprocessed_dir=$1  # working diretory
    local poly_a_file=$2  # name of poly_a file
    local local_dir=$3  # path to local_dir

    awk -v OFS='\t' '{print "chr"$1,$2,$3,"polya_"$4_$10}' ${unprocessed_dir}/${poly_a_file} \
        > ${local_dir}/polyasites_filtered_hg38.bed
}


# =============================================================================
# Genomic variant hotspots from Long & Xue, Human Genomics, 2021. First, expand
# hotspot clusers to their individual mutation type. Then, file is split into
# CNVs, indels, and SNPs.
# Arguments:
#   $1 - name of hotspot file
#   $1 - name of hotspot file
#   $2 - path to liftOver and liftover chain
#   $3 - directory to save parsed files
# =============================================================================
function _var_hotspots () {
    local unprocessed_dir=$1
    local var_file=$2
    local liftover_dir=$3
    local local_dir=$4

    tail -n +2 ${unprocessed_dir}/${var_file} \
        | awk '$4 == "Cluster"' \
        | sed 's/(/\t/g' \
        | cut -f1,2,3,6 \
        | sed -e 's/; /,/g' -e 's/)//g' \
        | bedtools expand -c 4 \
        | cat - <(awk -v OFS='\t' '$4 == "GV hotspot"' ${unprocessed_dir}/${var_file} | cut -f1,2,3,5) \
        | sort -k 1,1 -k2,2n > ${unprocessed_dir}/hotspots_expanded_hg18.bed

    # liftover to hg38
    _liftover_19_to_38 \
        ${liftover_dir} \
        ${unprocessed_dir} \
        hotspots_expanded_hg18

    for variant in CNV SID SNP; do
        local varlower=$(echo $variant | tr '[:upper:]' '[:lower:]') # lowercasing
        awk -v variant=$variant '$4 ~ variant' ${unprocessed_dir}/hotspots_expanded_hg18.lifted_hg38.bed \
            > ${local_dir}/${varlower}_hg38.bed
        if [[ $variant == SID ]]; then
            mv ${local_dir}/${varlower}_hg38.bed ${local_dir}/indels_hg38.bed
        fi
    done
}


# =============================================================================
# Replication hotspots from Long & Xue, Human Genomics, 2021. Each phase is
# split into a separate file for node attributes.
# =============================================================================
function _replication_hotspots () {
    local unprocessed_dir=$1
    local var_file=$2
    local liftover_dir=$3
    local local_dir=$4

    awk -v string='Replication' '$4 ~ string' ${unprocessed_dir}/${var_file} \
        | awk -v OFS='\t' '{print $1, $2, $3, "rep"$6}' \
        | sed 's/^/chr/g' \
        | tr '[:upper:]' '[:lower:]' \
        > ${unprocessed_dir}/rep_formatted_hg18.bed 

    # liftover to hg38
    _liftover_19_to_38 \
        ${liftover_dir} \
        ${unprocessed_dir} \
        rep_formatted_hg18

    # # print out a file per phase
    # awk '{file=$4 ".txt"; print > file}' ${unprocessed_dir}/rep_formatted_hg18.lifted_hg38.bed

    # print out a file per unique value in the fourth column
    awk -v output_dir="$local_dir" '{
        file=sprintf("%s/%s_hg38.bed", output_dir, $4);
        print > file
    }' "${unprocessed_dir}/rep_formatted_hg18.lifted_hg38.bed"

    # # symlink files to local directory
    # for phase in repg1b repg2 reps1 reps2 reps3 reps4;
    # do
    #     ln -s ${unprocessed_dir}/${phase}.txt ${local_dir}/${phase}_hg38.bed
    # done
}


# =============================================================================
# Average recombination rate from deCODE - Halldorsson et al, Science, 2019.
# BigWig is converted to Wig then to .bed via BEDOPS.
# =============================================================================
function _recombination () {
    local reference_dir=$1
    local unprocessed_dir=$2
    local recomb_file=$3
    local local_dir=$4

    ${reference_dir}/bigWigToWig \
        ${unprocessed_dir}/${recomb_file}.bw \
        ${unprocessed_dir}/${recomb_file}.wig

    wig2bed < ${unprocessed_dir}/${recomb_file}.wig \
        > ${unprocessed_dir}/${recomb_file}.bed

    awk -v OFS="\t" '{print $1, $2, $3, $5, $4}' ${unprocessed_dir}/${recomb_file}.bed \
        > ${local_dir}/recombination_hg38.bed
}


# =============================================================================
# Simple CpGisland parser to remove the first column and make the fourth column
# chr_star_cpgisland
# =============================================================================
function _cpg_islands () {
    local unprocessed_dir=$1
    local cpg_file=$2
    local local_dir=$3

    awk -v OFS="\t" '{print $2, $3, $4, $1"_"$2"_cpgisland"}' ${unprocessed_dir}/${cpg_file} \
        > ${local_dir}/cpgislands_hg38.bed
}


# =============================================================================
# Parser to split reference repeatmasker calls into LINE, LTR, SINE, and RNA
# repeat files. (LTR includes retroposons)
# https://repeatmasker.org/genomes/hg38/RepeatMasker-rm405-db20140131/hg38.fa.out.gz
# =============================================================================
function _repeatmasker () {
    local unprocessed_dir=$1
    local repeat_file=$2
    local local_dir=$3

    awk -v OFS="\t" '{print $6, $7, $8, $12}' ${unprocessed_dir}/${repeat_file} \
        | awk -v OFS="\t" -v dir="${local_dir}" '
            {
                match_type = "";
                file_suffix = "_hg38.bed";
                if ($4 == "LINE") { match_type = "line" }
                else if ($4 == "LTR" || $4 == "Retroposon") { match_type = "ltr" }
                else if ($4 == "SINE") { match_type = "sine" }
                else if ($4 ~ /RNA/) { match_type = "rnarepeat" }
                
                if (match_type != "") {
                    output_file = dir "/" match_type file_suffix;
                    print $1, $2, $3, $4 > output_file;
                }
            }
        '
}



# =============================================================================
# Parser to filter the phastcon file to only include regions that pass a certain
# score. We first calculate the mean score of each region, then filter the
# region based on the input score.
# =============================================================================
function _phastcons () {
    local reference_dir=$1
    local unprocessed_dir=$2
    local phastcon_file=$3
    local local_dir=$4
    local score_cutoff=$5

    # Convert phastcon bigwig to bedGraph
    ${reference_dir}/bigWigToBedGraph \
        ${unprocessed_dir}/${phastcon_file} \
        ${unprocessed_dir}/phastcons.bedGraph

    # Grab areas with a score greater than 0.7
    awk -v OFS="\t" -v score_cutoff=$score_cutoff '$4 >= score_cutoff' ${unprocessed_dir}/phastcons.bedGraph \
        | bedtools merge -i - \
        | awk -v OFS="\t" '{print $1, $2, $3, "conserved"}' \
        > ${local_dir}/phastcons_hg38.bed
}


# =============================================================================
# Main function! Takes care of all the data processing and downloading.
# =============================================================================
function main () {
    log_progress "Setting up directories and inputs"
    local root_dir=$1
    local unprocessed_dir="${root_dir}/unprocessed"
    local shared_dir="${root_dir}/shared_data"
    local local_dir="${shared_dir}/local"
    local reference_dir="${shared_dir}/references"

    log_progress "Downloading and parsing regulatory elements..."
    declare -A files_to_download=(
        ["$unprocessed_dir/40246_2021_318_MOESM3_ESM.txt"]="https://static-content.springer.com/esm/art%3A10.1186%2Fs40246-021-00318-3/MediaObjects/40246_2021_318_MOESM3_ESM.txt"
        ["$unprocessed_dir/recombAvg.bw"]="https://hgdownload.soe.ucsc.edu/gbdb/hg38/recombRate/recombAvg.bw"
        ["$unprocessed_dir/cpgIslandExt.txt.gz"]="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cpgIslandExt.txt.gz"
        ["$unprocessed_dir/simpleRepeat.txt.gz"]="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/simpleRepeat.txt.gz"
        ["$unprocessed_dir/atlas.clusters.2.0.GRCh38.96.bed.gz"]="https://polyasite.unibas.ch/download/atlas/2.0/GRCh38.96/atlas.clusters.2.0.GRCh38.96.bed.gz"
        ["$unprocessed_dir/microsat.txt.gz"]="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/microsat.txt.gz"
        ["$unprocessed_dir/rmsk.txt.gz"]="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/rmsk.txt.gz"
        ["$unprocessed_dir/hg38.phastCons30way.bw"]="http://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons30way/hg38.phastCons30way.bw"
    )

    # download each file
    for target_path in "${!files_to_download[@]}"; do
        _download_raw_file "$target_path" "${files_to_download[$target_path]}"
    done

    # gunzip the .gz files
    for file in ${unprocessed_dir}/simpleRepeat.txt.gz ${unprocessed_dir}/cpgIslandExt.txt.gz ${unprocessed_dir}/microsat.txt.gz ${unprocessed_dir}/rmsk.txt.gz; do
        gunzip -k $file
    done

    # filter polyA file
    log_progress "Processing poly_a tail binding sites..."
    _poly_a \
        ${unprocessed_dir} \
        atlas.clusters.2.0.GRCh38.96.bed \
        ${local_dir}

    # expand the variant hotspots
    log_progress "Processing genomic variant hotspots..."
    _var_hotspots \
        ${unprocessed_dir} \
        40246_2021_318_MOESM3_ESM.txt \
        ${reference_dir} \
        $local_dir

    # replication hotspots
    log_progress "Processing replication hotspots..."
    _replication_hotspots \
        ${unprocessed_dir} \
        40246_2021_318_MOESM3_ESM.txt \
        ${reference_dir} \
        $local_dir

    # get average recombination rate
    log_progress "Processing recombination rate..."
    _recombination \
        ${reference_dir} \
        ${unprocessed_dir} \
        recombAvg \
        ${local_dir}

    # CpG islands
    log_progress "Processing CpG islands..."
    _cpg_islands \
        ${unprocessed_dir} \
        cpgIslandExt.txt \
        ${local_dir}

    # Microsatellites
    log_progress "Parse microsatellites..."
    awk -v OFS="\t" '{print $2, $3, $4, $5}' ${unprocessed_dir}/microsat.txt \
        > ${local_dir}/microsatellites_hg38.bed

    # Simple repeats
    log_progress "Parse simple repeats..."
    awk -v OFS="\t" '{print $2, $3, $4, $5}' ${unprocessed_dir}/simpleRepeat.txt \
        > ${local_dir}/simple_repeats_hg38.bed

    # Repeat masker
    log_progress "Parse repeat masker into individual repeat types."
    _repeatmasker \
        ${unprocessed_dir} \
        "rmsk.txt" \
        ${local_dir}

    # Conservation scores
    log_progress "Parse phastCons scores..."
    _phastcons \
        ${reference_dir} \
        ${unprocessed_dir} \
        "hg38.phastCons30way.bw" \
        ${local_dir} \
        0.7
}


# =============================================================================
# Run main function!
# =============================================================================
main $root_directory