#!/bin/bash
#
# Script to download data from the tissue-specific data. The script downloads
# methylation and miRNA data from the ENCODE portal, tf binding footprints from
# Vierstra et al., and superenhancers from the super enhancer database (SeDB).
#
# To run the script:
# $ bash tissue_specific_datasets.sh
# $     --root_directory <root_directory>
# 
# The script will add the following directory (with asterisks indicating)
# root_directory
# ├── unprocessed
# ├── raw_tissue_data
# │   └── *tissue
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
# Setting up variables to track progress
# =============================================================================
log_progress() {
    echo -e "[$(date +%Y-%m-%dT%H:%M:%S%z)] "
}

# =============================================================================
# Function to download ENCODE datasets from the portal. The final entry in each
# array will be the TSV miRNA file, and any preceding accessions will be CpG
# methylation beds. Each file is downloaded then unzipped to the tissue-specific
# directory.
# =============================================================================
function _encode_downloader () {
    declare -A encode_map
    encode_map["k562"]="ENCFF660IHA ENCFF721EFA"
    encode_map["imr90"]="ENCFF404JRI ENCFF221TXA"
    encode_map["gm12878"]="ENCFF570TIL ENCFF343RGE"
    encode_map["hepg2"]="ENCFF690FNR ENCFF671ZNH"
    encode_map["h1_esc"]="ENCFF434CNG ENCFF851ANC"
    encode_map["hmec"]="ENCFF478RLX ENCFF164YOM"
    encode_map["nhek"]="ENCFF679KWT"
    encode_map["hippocampus"]="ENCFF520TSX"
    encode_map["lung"]="ENCFF532PDE ENCFF453HAD ENCFF821OMH ENCFF434PBY"
    encode_map["pancreas"]="ENCFF556VBV ENCFF080TDH ENCFF689HVV ENCFF296NWQ"
    encode_map["skeletal_muscle"]="ENCFF341TLB ENCFF796CSP ENCFF939WLM ENCFF317BBO"
    encode_map["small_intestine"]="ENCFF387UUA ENCFF891UDL ENCFF113RNL ENCFF538RSM"
    encode_map["liver"]="ENCFF080ETP ENCFF416CUZ ENCFF613KZJ"
    encode_map["aorta"]="ENCFF669KDL ENCFF901HLR ENCFF622GIX"
    encode_map["skin"]="ENCFF520YPF ENCFF649IEY ENCFF978LIV ENCFF625JKM"
    encode_map["left_ventricle"]="ENCFF908SEW ENCFF318IKY ENCFF398ERY ENCFF119HDC"
    encode_map["mammary"]="ENCFF478RLX ENCFF164YOM"
    encode_map["spleen"]="ENCFF287TJO ENCFF527IVK ENCFF336UWV ENCFF564OCC"
    encode_map["ovary"]="ENCFF889UGC ENCFF716SXG ENCFF454NPL ENCFF087HIG"
    encode_map["adrenal"]="ENCFF942KJO ENCFF774YCZ ENCFF748OTV ENCFF817OKW"

    local raw_tissue_dir=$1
    local unprocessed_dir=$2
    local base_url="https://www.encodeproject.org/files"

    for tissue_name in "${!encode_map[@]}"; do
        mkdir -p "${raw_tissue_dir}/${tissue_name}"

        # read accession and get last accession index (the tsv)
        read -ra accessions <<< "${encode_map[$tissue_name]}"
        local last_index=$((${#accessions[@]} - 1))

        # download accessions!
        for i in "${!accessions[@]}"; do
            local accession="${accessions[$i]}"

            if [ "$i" -eq "$last_index" ]; then
                local file_extension=".tsv"
                local output_path="${raw_tissue_dir}/${tissue_name}/${accession}${file_extension}"
            else
                local file_extension=".bed.gz"
                local output_path="${unprocessed_dir}/${accession}${file_extension}"
            fi
            
            local url="${base_url}/${accession}/@@download/${accession}${file_extension}"

            if [ -f "${output_path}" ]; then
                log_progress "File already exists: ${output_path}"
                continue
            else
                log_progress "Downloading ${url} to ${output_path}"
                wget -O "${output_path}" "${url}"
            fi

            # if file is a .gz, unzip it to the tissue directory
            if [[ "${output_path}" == *.gz ]]; then
                gunzip -c "${output_path}" > "${raw_tissue_dir}/${tissue_name}/${accession}.bed"
                rm "${output_path}" # Optionally remove the .gz file after extraction
                echo "Unzipped ${output_path} to ${raw_tissue_dir}/${tissue_name}/${accession}.bed"
            fi
        done
    done
}


# =============================================================================
# Function to download hippocampus data, convert to bed, and merge CpG sites. It
# is the only tissue in this dataset that does not have an ENCODE methylation
# accession, and is processed here separately.
# =============================================================================
_hippocampus_data() {
    local unprocessed_dir=$1
    local tissue_dir=$2

    # Download hippocampus methylation data from GSE and convert to bed
    wget -O "${unprocessed_dir}/GSM916050_BI.Brain_Hippocampus_Middle.Bisulfite-Seq.150.wig.gz" \
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM916nnn/GSM916050/suppl/GSM916050%5FBI%2EBrain%5FHippocampus%5FMiddle%2EBisulfite%2DSeq%2E150%2Ewig%2Egz"

    gunzip "${unprocessed_dir}/GSM916050_BI.Brain_Hippocampus_Middle.Bisulfite-Seq.150.wig.gz"

    # convert to bed
    wig2bed < "${unprocessed_dir}/GSM916050_BI.Brain_Hippocampus_Middle.Bisulfite-Seq.150.wig" \
    > "${unprocessed_dir}/GSM916050_BI.Brain_Hippocampus_Middle.Bisulfite-Seq.150.bed"

    # keep columns 1-4 and only rows where col4 >= .80
    awk -vOFS="\t" '$5 >= 0.80 {print $1, "$2, "$3, $5}' \
    "${unprocessed_dir}/GSM916050_BI.Brain_Hippocampus_Middle.Bisulfite-Seq.150.bed" \
    | bedtools merge -i - \
    > "${tissue_dir}/hippocampus/GSM916050_BI.Brain_Hippocampus_Middle.Bisulfite-Seq.150.merged.bed"
}


# =============================================================================
# Download tfbinding footprints from Vierstra et al. and save to the tissue
# directory.
# =============================================================================
_tf_binding_footprints() {
    local tissue_dir=$1

    declare -A tissue_links=(
        ["k562"]="https://resources.altius.org/~jvierstra/projects/footprinting.2020/per.dataset/h.K562-DS52908/interval.all.fps.0.0001.bed"
        ["imr90"]="https://resources.altius.org/~jvierstra/projects/footprinting.2020/per.dataset/IMR90-DS13219/interval.all.fps.0.0001.bed"
        ["gm12878"]="https://resources.altius.org/~jvierstra/projects/footprinting.2020/per.dataset/GM12865-DS12436/interval.all.fps.0.0001.bed"
        ["hepg2"]="https://resources.altius.org/~jvierstra/projects/footprinting.2020/per.dataset/h.HepG2-DS24838/interval.all.fps.0.0001.bed"
        ["h1_esc"]="https://resources.altius.org/~jvierstra/projects/footprinting.2020/per.dataset/h.ESC.H9-DS39598/interval.all.fps.0.0001.bed"
        ["hmec"]="https://resources.altius.org/~jvierstra/projects/footprinting.2020/per.dataset/MCF10a_ER_SRC-DS22980/interval.all.fps.0.0001.bed"
        ["nhek"]="https://resources.altius.org/~jvierstra/projects/footprinting.2020/per.dataset/Skin_keratinocyte-DS18692/interval.all.fps.0.0001.bed"
        ["hippocampus"]="https://resources.altius.org/~jvierstra/projects/footprinting.2020/per.dataset/h.brain-DS23541/interval.all.fps.0.0001.bed"
        ["lung"]="https://resources.altius.org/~jvierstra/projects/footprinting.2020/per.dataset/NHLF-DS12829/interval.all.fps.0.0001.bed"
        ["pancreas"]="https://resources.altius.org/~jvierstra/projects/footprinting.2020/per.dataset/h.ISL1-DS55938/interval.all.fps.0.0001.bed"
        ["skeletal_muscle"]="https://resources.altius.org/~jvierstra/projects/footprinting.2020/per.dataset/HSMM-DS14426/interval.all.fps.0.0001.bed"
        ["small_intestine"]="https://resources.altius.org/~jvierstra/projects/footprinting.2020/per.dataset/h.CEC-DS37513/interval.all.fps.0.0001.bed"
        ["liver"]="https://resources.altius.org/~jvierstra/projects/footprinting.2020/per.dataset/h.hepatocytes-DS32057/interval.all.fps.0.0001.bed"
        ["aorta"]="https://resources.altius.org/~jvierstra/projects/footprinting.2020/per.dataset/AoAF-DS13513/interval.all.fps.0.0001.bed"
        ["skin"]="https://resources.altius.org/~jvierstra/projects/footprinting.2020/per.dataset/Skin_keratinocyte-DS18692/interval.all.fps.0.0001.bed"
        ["left_ventricle"]="https://resources.altius.org/~jvierstra/projects/footprinting.2020/per.dataset/HCF-DS12501/interval.all.fps.0.0001.bed"
        ["mammary"]="https://resources.altius.org/~jvierstra/projects/footprinting.2020/per.dataset/HMF-DS13368/interval.all.fps.0.0001.bed"
    )

    for tissue in "${!tissue_links[@]}"; do
        wget -P ${tissue_dir}/${tissue} \
        "${tissue_links[$tissue]}"
    done
}


# =============================================================================
# Download the coordinations of DHSs with evidence of being bound by TFs from
# Meuleman, et al 2020. We concatenate the files, keeping the first 3 columns,
# and adding the TF name as the 4th column.
# =============================================================================


# =============================================================================
# Download superenhancers from SeDB and save to the tissue directory.
# =============================================================================
_sedb() {
    local tissue_dir=$1
    url_prefix="https://bio.liclab.net/sedb/download/new/SE_bed/hg38/"

    declare -A super_enhancers=(
        ["k562"]="SE_01_0039"
        ["imr90"]="SE_02_0497"
        ["gm12878"]="SE_01_0030"
        ["hepg2"]="SE_01_0038"
        ["h1_esc"]="SE_00_0012"
        ["hmec"]="SE_02_0056"
        ["nhek"]="SE_02_0216"
        ["hippocampus"]="SE_02_0351"
        ["lung"]="SE_02_1297"
        ["pancreas"]="SE_00_0030"
        ["skeletal_muscle"]="SE_01_0063"
        ["small_intestine"]="SE_00_0049"
        ["liver"]="SE_02_1299"
        ["aorta"]="SE_00_0004"
        ["skin"]="SE_01_0041"
        ["left_ventricle"]="SE_00_0014"
        ["mammary"]="SE_01_0045"
        ["spleen"]="SE_01_0084"
        ["ovary"]="SE_00_0029"
        ["adrenal"]="SE_01_0069"
    )

    for tissue in "${!super_enhancers[@]}"; do
        wget -P ${tissue_dir}/${tissue} \
        "${url_prefix}${super_enhancers[$tissue]}_SE_hg38.bed"
    done
}


# =============================================================================
# run main_func function! 
# =============================================================================
main () {
    local root_dir=$1
    local tissue_dir="${root_dir}/raw_tissue_data"
    local unprocessed_dir="${root_dir}/unprocessed"

    _encode_downloader \
        "${tissue_dir}" \
        "${unprocessed_dir}"

    # Download and process methylation data for hippocampus, keeping CpG sites
    # with >= 80% methylated reads
    _hippocampus_data "${unprocessed_dir}" "${tissue_dir}"

    # download NHEK CpG data
    wget \
        -P \
        "${root_dir}/raw_tissue_data/nhek" \
        https://egg2.wustl.edu/roadmap/data/byDataType/dnamethylation/WGBS/FractionalMethylation_bigwig/E058_WGBS_FractionalMethylation.bigwig

    # download tf binding footprints
    _tf_binding_footprints "${tissue_dir}"

    # download superenhancers
    _sedb "${tissue_dir}"
}


# =============================================================================
# run main function!
# =============================================================================
main "${root_directory}"
# echo "Total time: $(convertsecs SECONDS)"