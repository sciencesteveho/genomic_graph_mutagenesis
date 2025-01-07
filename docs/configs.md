
# Omics Graph Learning Configs


## Experiment Config

See `ogl/configs/experiments` for examples.

| Parameter                      | Description                                                                                                                                                                                                                           |
|--------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `attribute_references`| These are reference attribute tracks for informing node feature data and are provided. Place them in `ref_dir`. |
| `baseloops`| The suffix for the 3D chromatin contacts used as graph basis, e.g. if `allcontacts`, then the loop file is expected to be in `root_dir/shared_data/processed_loops/allcontacts/*sample*_contacts.bedpe`  |
| `blacklist` | Full path to the ENCODE blacklist v2 bed. |
| `chromfile`  | Full path to an h38 chromosome size file. |
| `config_dir` | Full path to where configs for OGL will be stored. |
| `derived_directories`  | These are directories deriving from the root directory. Create the directory structure as in the README and leave these untouched. |
| `differentiate_tf`                     | Add a one-hot encoded feature to differentiate between transcription factors and their genes.  |
| `experiment_name`                     | Name of the experiment (user specified).  |
| `fasta`                     | Full path to an hg38 fasta file.  |
| `feat_window`                     | Size of the window for which edges get drawn between two features in linear proximity. Default is `6250` base pairs.  |
| `gene_gene`                     | Add gene -> gene interactions if they are linked via 3D chromatin data.  |
| `interaction_types`                     | Types of interaction data appended to the graph structure.  |
| `liftover`                     | Full path to UCSC liftover tool.  |
| `liftover_chain`                     | Full path to UCSC liftover chain (hg19 to hg38).  |
| `log_transform`                     | Log transformation to use on regression targets. Choices are `log2`, `log1p`, and `log10`.  |
| `nodes`                     | Additional node types beyond genes and regulatory elements to consider.  |
| `positional_encoding`                     | Whether to build and use positional encodings during GNN training. `train_positional_encoding` will be overridden by the argument parser.  |
| `regulatory_schema`                     | Specify regulatory element catalogues.  |
| `root_dir`                     | Full path to root directory, e.g. `path/to/graph_processing/`.  |
| `tissues`                     | List of samples for the model, each of which requires a sample config. OGL can train multiple samples concurrently.  |
| `training_targets`                     | Full path to expression matrices for target assemblage. See [model_target_data.py](../programmatic_data_download/model_target_data.py) for pre-processing scripts.  |
| `test_chrs`                     | Whole-chromosome holdouts to use for testing. If unspecified, a random subset of 10% of the targets will be used.  |
| `val_chrs`                     | Whole-chromosome holdouts to use for validation. If unspecified, a random subset of 10% of the targets will be used.  |
---

## Sample Config

See `ogl/configs/samples` for examples.

| Parameter                   | Description                                                                                                                                                                                                                                           |
|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `root_dir` | Full path to root dir *exclusive* of `graph_processing`.  |
| `features` | Tissue-specific node feature tracks. These should be in `root_dir/graph_processing/raw_tissue_data/*sample*`. See [merge_epimap_bedgraphs.py](../programmatic_data_download/merge_epimap_bedgraphs.py) for pre-processing script.   |
| `interaction` | Resources used during graph construction for interaction data.  |
| `local` | Genome-static node feature tracks. These should be in `root_dir/graph_processing/shared_data/local`. Pre-processing scripts are available [here](../programmatic_data_download/)  |
| `methylation` | Name of methylation file which should be placed in `root_dir/graph_processing/raw_tissue_data/*sample*`. Possible `cpg_filetypes` are `ENCODE` and `roadmap`. `cpg_liftover` specifies using the liftover chain if your CpG file is on hg19.  |
| `resources` | Full paths to specific resources used for graph construction. Reference files are provided.  |
| `tissue_specific_nodes` | Tissue-specific datasets for graph parsing. These include the cis-regulatory module file, a super enhancer track, transcription-factor binding fingerprints, and 3D chromatin files. |
