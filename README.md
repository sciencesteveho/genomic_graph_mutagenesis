# Genomic Graph Mutagenesis
Tools to construct graphs heterogenous multi-omics data and train a GNN to regress values of gene expression and protein abundance. Graphs are mutagenized to query the impact of individual features on biological function.
&nbsp;

<div align="center">
    <img src='docs/_static/placeholder.png'>
</div>
&nbsp;

## Installation

```sh
$ git clone https://github.com/sciencesteveho/genomic_graph_mutagenesis.git
```

&nbsp;

## Dependencies


```sh
cmapPy==4.0.1
joblib==1.0.1
keras==2.10.0
Keras-Preprocessing==1.1.2
MACS2==2.2.8
networkx==2.6.3
numpy==1.20.2
nvidia-cudnn-cu11==8.5.0.96
pandas==1.2.4
pybedtools==0.9.0
pysam==0.19.0
PyYAML==5.4.1
scikit-learn==0.24.2
scipy==1.7.3
shyaml==0.6.2
tensorflow==2.10.0
torch==1.13.1
torch-geometric==2.3.0
tqdm==4.60.0
```
Additionally, GGM uses peakMerge.py from ReMap2022 (Hammal et al., *Nucleic Acids Research*, 2021) to call tissue-specific cis-regulatory modules from epimap data. Download the script and place its path in the configuration file.
```sh
wget https://raw.githubusercontent.com/remap-cisreg/peakMerge/main/peakMerge.py
```
&nbsp;

## Usage

Note: not all arguments are compatible with one another, so see examples below for the program's capabilities.
```sh
# Convert epimap bigwig files to broad and narrow peaks
for tissue in hippocampus left_ventricle liver lung mammary pancreas skeletal_muscle skin small_intestine;
do
    sbatch merge_epimap.sh $tissue
done

# Add chromatin loops together
sh chrom_loops_basefiles.sh

# Preparse bedfiles
sh preparse.sh

# Run python scripts
python -u genomic_graph_mutagenesis/prepare_bedfiles.py --config ${yaml}
python -u genomic_graph_mutagenesis/edge_parser.py --config ${yaml}
python -u genomic_graph_mutagenesis/local_context_parser.py --config ${yaml}
python -u genomic_graph_mutagenesis/graph_constructor.py --config ${yaml}

sh concat nx.sh 

# Train graph neural network

# Mutagenize 

# Plotting and visualization
```
&nbsp;

# Tissue-specific models
Base interactions are derived tissue-specific chromatin loops, which is then combined with the interaction type graphs to creates the base nodes. Edges are added to these base nodes if local context nodes are within 2mb of a base node.

There are 14 node types, 4 edge types, and each node has a 36-dimensional feature vector.
Each training target is a 4-dimensional feature vector.

```
The following features have node representations:
    Tissue-specific
        Genes (GENCODE, PPI interactions from IID)
        TFs (Marbach, TFMarker)
        MicroRNAs (mirDIP for tissue-specific, miRTarBase for interactions)
        Cis-regulatory modules (peakMerge against epimap narrow peaks)
        Transcription factor binding sites (Vierstra et al., Nature, 2020)
        TADs
        Super-enhancers (sedb)

    Genome-static
        Cpgislands
        Gencode (genes)
        Promoters (overlap between encode cCREs and epimap)
        Enhancers (overlap between encode cCREs and epimap)
        Dyadic elements (overlap between encode cCREs and epimap)
        CTCF-cCRE (encode cCRE)
        Transcription start sites

The following are represented as attributes:
    Tissue-specific
        CpG methylation

        Peak calls
            DNase
            ATAC-seq
            CTCF
            H3K27ac
            H3K27me3
            H3K36me3
            H3K4me1
            H3K4me2
            H3K4me3
            H3K79me2
            H3K9ac
            H3K9me3
            POLR2A
            RAD21
            SMC3

    Genome-static
        GC content
        Microsatellites
        Conservation (phastcons)
        Poly(a) binding sites (overlap)
        LINEs (overlap)
        Long terminal repeats (overlap)
        Simple repeats (overlap)
        SINEs (overlap)
        Hotspots
            snps
            indels
            cnvs 
        miRNA target sites
        RNA binding protein binding sites
        Replication phase
            g1b
            g2
            s1
            s3
            s4
            s4
        Recombination rate (averaged)
```

## Working Tissues
```
    Hippocampus
    Left ventricle
    Liver
    Lung
    Mammary
    Pancreas
    Skeletal muscle
    Skin
    Small intestine
```
## Cell type models (future)
```
    HeLa
    K562
    Neural progenitor cell
```
&nbsp;


# Universal Genome
The universal genome annotation borrows the core ideas from the tissue-specific models but utilizes data sources that represent large-scale data collected from multiple genomes. Additionally, the universal genome model has an additional node type, DNAse-hypersensitivty sites.

There are 15 node types, 4 edge types, and each node has a 779-dimensional feature vector. Each training target is a 120-dimensional feature vector.

```
Chromatin loops are variable chromatin loops derived from Grubert at al.
DHS nodes are all 3.59 million DNase I sites from Meuleman et al., Nature, 2020.

Protein-protein interactions come from the human_only set from the Integrated Interactions Database.
TF-Gene interactions are the combinded top 20% interactions across 24 models from Marbach et al.

MicroRNAs are the entirety of miRTarBase interactions without tissue-specific filtering with mirDIP.
Cis-regulatory are modules Homo Sapiens CRMs from ReMap2022.
Transcription factor binding sites come from (Vierstra et al., Nature, 2020)
TADs are the scaffolds from TadMap by Singh and Berger, 2021.
Super-enhancers are all human super enhancers from the SEdb.

Genome-static nodes are kept as they are in tissue-specific models
CpG methlylation is represented as a 37n vector, each representing a different reference genome derived from the roadmap project. CpGs are merged.
Epigenetic are represented as a 722n vector, where each track is the average signal in a different tissue.