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
# >>> bash regulatory_element_assemblage.sh \
# >>>   --root_directory /path/to/your/root/directory
#


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
function _liftover_elements_19_to_38 () {
    local liftover_dir="$1"  # path/to/liftover/directory
    local unprocessed_dir="$2"  # path/to/unprocessed/directory
    local output_dir="$3"  # path/to/output/directory
    local file_name="$4"  # name of file to liftover
    local element_name="$5"  # naming convention for the regulatory element

    ${liftover_dir}/liftOver \
        ${unprocessed_dir}/${file_name}.bed \
        ${liftover_dir}/hg19ToHg38.over.chain.gz \
        ${unprocessed_dir}/${file_name}.lifted_hg38.bed \
        ${unprocessed_dir}/${file_name}.unlifted

    awk -v OFS='\t' -v ename="${element_name}" '{print $1,$2,$3,$1"_"$2"_"ename}' \
        ${unprocessed_dir}/${file_name}.lifted_hg38.bed \
        > ${output_dir}/${file_name}.lifted_hg38.bed
}


# =============================================================================
# Function to overlap SCREEN regulatory regions with EpiMap regulatory regions.
# Done for both enhancers and promoters.
# =============================================================================
function _overlap_regulatory_regions () {
    local reg_dir=$1  # path/to/your working directory
    local epimap_file=$2  # name of lifted epimap file
    local encode_file=$3  # name of encode file
    local element_name=$4  # naming convention for the regulatory element

    bedtools intersect \
        -a "${reg_dir}/${encode_file}" \
        -b "${reg_dir}/${epimap_file}" \
        -wa \
        -wb \
        | sort -k1,1 -k2,2n \
        | cut -f1,2,3,6 \
        | cut -f1 -d',' \
        | uniq \
        | awk -v OFS='\t' -v ename="${element_name}" '{print $1,$2,$3,$1"_"$2"_"ename}' \
        > "${reg_dir}/${element_name}_epimap_screen_overlap.bed"
}


# =============================================================================
# Function to overlap SCREEN regulatory regions with EpiMap dyadic elements.
# Because dyadic elements can act as both enhancers and promoters, an overlap is
# performed against both.
# =============================================================================
function _overlap_dyadic_elements () {
    local reg_dir=$1  # path/to/your working directory
    local dyadic_file=$2  # name of lifted dyadic file
    local enhancer_file=$3  # name of SCREEN enhancer file
    local promoter_file=$4  # name of SCREEN promoter file

    bedtools intersect \
        -a ${reg_dir}/${dyadic_file} \
        -b ${reg_dir}/${enhancer_file} ${reg_dir}/${promoter_file} \
        -wa \
        | sort -k1,1 -k2,2n \
        | cut -f1,2,3 \
        | uniq \
        | awk -v OFS='\t' '{print $1,$2,$3,$1"_"$2"_dyadic"}' \
        > ${reg_dir}/dyadic_epimap_screen_overlap.bed
}


# =============================================================================
# Function to remove any duplicate elements from the catalogues because of the
# overlap against the dyadic elements.
# =============================================================================



# =============================================================================
# Create node attr references for the different regulatory catalogues. These
# will be used later after the edge parser to place each node. The names are
# hardcoded and meant to only be run after the rest of the script is completed
# without error.
# =============================================================================
function _make_ref_for_regulatory_elements () {
    local reg_dir=$1  # path/to/your working directory
    local reference_dir=$2  # path/to/reference/directory

    # encode only
    cat \
        ${reg_dir}/GRCh38-ELS.bed \
        ${reg_dir}/GRCh38-PLS.bed \
        | sort -k1,1 -k2,2n \
        | uniq \
        | sort -k1,1 -k2,2n \
        > ${reference_dir}/regulatory_elements_encode_node_attr.bed

    # epimap only
    cat \
        ${reg_dir}/ENH_masterlist_locations.lifted_hg38.bed \
        ${reg_dir}/PROM_masterlist_locations.lifted_hg38.bed \
        ${reg_dir}/DYADIC_masterlist_locations.lifted_hg38.bed \
        | sort -k1,1 -k2,2n \
        | uniq \
        | sort -k1,1 -k2,2n \
        > ${reference_dir}/regulatory_elements_epimap_node_attr.bed

    # intersect
    cat \
        ${reg_dir}/dyadic_epimap_screen_overlap.bed \
        ${reg_dir}/enhancer_epimap_screen_overlap.bed \
        ${reg_dir}/promoter_epimap_screen_overlap.bed \
        | sort -k1,1 -k2,2n \
        | uniq \
        | sort -k1,1 -k2,2n \
        > ${reference_dir}/regulatory_elements_intersect_node_attr.bed

    # union
    cat \
        ${reg_dir}/DYADIC_masterlist_locations.lifted_hg38.bed \
        ${reg_dir}/enhancer_all_union_hg38.bed \
        ${reg_dir}/promoter_all_union_hg38.bed \
        | sort -k1,1 -k2,2n \
        | uniq \
        | sort -k1,1 -k2,2n \
        > ${reference_dir}/regulatory_elements_union_node_attr.bed
}


# =============================================================================
# ENCODE SCREEN CTCF-only cCREs from SCREEN V3
# Arguments:
#   $1 - path to ctcf file
#   $2 - cCRE file
#   $3 - directory to save parsed files
# =============================================================================
function _screen_ctcf () {
    local unprocessed_dir=$1
    local ctcf_file=$2
    local output_dir=$3

    grep CTCF-only ${unprocessed_dir}/${ctcf_file} \
        | awk -v OFS='\t' '{print $1,$2,$3,"ctcfccre"}' \
        > ${output_dir}/ctcfccre_parsed_hg38.bed
}


# =============================================================================
# Main function! Takes care of all the data processing and downloading.
# =============================================================================
function main () {
    log_progress "Setting up directories and inputs"
    local root_dir=$1
    local reg_dir=${root_dir}/shared_data/regulatory_elements
    local local_dir=${root_dir}/shared_data/local
    local reference_dir=${root_dir}/shared_data/references
    local unprocessed_dir=${reg_dir}/unprocessed

    log_progress "Downloading and parsing regulatory elements..."
    declare -A files_to_download=(
        ["$unprocessed_dir/GRCh38-PLS.bed"]="https://downloads.wenglab.org/cCREs/GRCh38-PLS.bed"
        ["$unprocessed_dir/GRCh38-ELS.bed"]="https://downloads.wenglab.org/cCREs/GRCh38-ELS.bed"
        ["$unprocessed_dir/GRCh38-CTCF.bed"]="https://downloads.wenglab.org/cCREs/GRCh38-CTCF.bed"
        ["$unprocessed_dir/DYADIC_masterlist_locations.bed"]="https://personal.broadinstitute.org/cboix/epimap/mark_matrices/DYADIC_masterlist_locations.bed"
        ["$unprocessed_dir/ENH_masterlist_locations.bed"]="https://personal.broadinstitute.org/cboix/epimap/mark_matrices/ENH_masterlist_locations.bed"
        ["$unprocessed_dir/PROM_masterlist_locations.bed"]="https://personal.broadinstitute.org/cboix/epimap/mark_matrices/PROM_masterlist_locations.bed"
    )

    # download each file
    for target_path in "${!files_to_download[@]}"; do
        _download_raw_file "$target_path" "${files_to_download[$target_path]}"
    done

    # Process SCREEN cCREs by adjusting naming convention
    log_progress "Processing SCREEN cCREs..."
    for file in GRCh38-ELS.bed GRCh38-PLS.bed; do
        awk -v OFS='\t' '{print $1,$2,$3,$1"_"$2"_"$4}' "${unprocessed_dir}/${file}" > "${reg_dir}/${file}"
    done

    # process SCREEN CTCF-only cCREs
    _screen_ctcf \
        "${unprocessed_dir}" \
        "GRCh38-CTCF.bed" \
        "${local_dir}"

    # map element name to file
    log_progress "Liftover EpiMap regulatory elements..."
    declare -A element_names=(
        ["DYADIC"]="dyadic"
        ["ENH"]="enhancer"
        ["PROM"]="promoter"
    )

    # liftover EpiMap regulatory elements using the associative array
    for key in "${!element_names[@]}"; do
        file="${key}_masterlist_locations"
        element="${element_names[$key]}"
        
        _liftover_elements_19_to_38 \
            "${reference_dir}" \
            "${unprocessed_dir}" \
            "${reg_dir}" \
            "${file}" \
            "${element}"
    done

    # get overlap catalogue
    log_progress "Overlapping regulatory elements..."
    _overlap_regulatory_regions \
        "${reg_dir}" \
        "ENH_masterlist_locations.lifted_hg38.bed" \
        "GRCh38-ELS.bed" \
        "enhancer"

    _overlap_regulatory_regions \
        "${reg_dir}" \
        "PROM_masterlist_locations.lifted_hg38.bed" \
        "GRCh38-PLS.bed" \
        "promoter"

    _overlap_dyadic_elements \
        "${reg_dir}" \
        "DYADIC_masterlist_locations.lifted_hg38.bed" \
        "GRCh38-ELS.bed" \
        "GRCh38-PLS.bed"

    # get union catalogue
    log_progress "Creating union catalogue..."
    cat \
        "${reg_dir}/enhancer_epimap_screen_overlap.bed" \
        "${reg_dir}/GRCh38-ELS.bed" \
        | sort -k1,1 -k2,2n \
        | awk -v OFS='\t' '{print $1,$2,$3,$1"_"$2"_enhancer"}' \
        > "${reg_dir}/enhancer_all_union_hg38.bed"

    cat \
        "${reg_dir}/promoter_epimap_screen_overlap.bed" \
        "${reg_dir}/GRCh38-PLS.bed" \
        | sort -k1,1 -k2,2n \
        | awk -v OFS='\t' '{print $1,$2,$3,$1"_"$2"_promoter"}' \
        > "${reg_dir}/promoter_all_union_hg38.bed"

    # make node attr references
    log_progress "Creating node attribute references..."
    _make_ref_for_regulatory_elements \
        "${reg_dir}" \
        "${reference_dir}"
}


# =============================================================================
# Run the main function!
# =============================================================================
main "${root_directory}"