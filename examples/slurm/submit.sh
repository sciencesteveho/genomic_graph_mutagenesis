#!/bin/bash

#SBATCH --job-name=submit
#SBATCH --mail-user=stevesho@umich.edu
#SBATCH --account=bio210019p
#SBATCH --mail-type=FAIL

# Number of nodes requested
#SBATCH --ntasks-per-node=1

# Partition
#SBATCH -p RM-shared

# Time
#SBATCH -t 1:00:00

# output to a designated folder
#SBATCH -o slurm_outputs/%x_%j.out

#echo commands to stdout
set -x

module load anaconda3/2022.10
conda activate /ocean/projects/bio210019p/stevesho/ogl




configs=(k562_allcontacts_global.yaml)
for config in "${configs[@]}"; do
  python ogl/omics_graph_learning/ogl_pipeline.py \
    --partition RM \
    --experiment_yaml ogl/configs/experiments/"${config}" \
    --target rna_seq \
    --model GAT \
    --gnn_layers 2 \
    --linear_layers 2 \
    --activation gelu \
    --dimensions 200 \
    --batch_size 32 \
    --learning_rate 0.00001 \
    --optimizer AdamW \
    --scheduler cosine \
    --dropout 0.3 \
    --residual distinct_source \
    --heads 2 \
    --positional_encoding
done

# configs=(all_celllines_alloopshicfdr.yaml)
# for config in "${configs[@]}"; do
#   python ogl/omics_graph_learning/ogl_pipeline.py \
#     --partition RM \
#     --experiment_yaml ogl/configs/experiments/"${config}" \
#     --target rna_seq \
#     --model GAT \
#     --gnn_layers 2 \
#     --linear_layers 2 \
#     --activation gelu \
#     --dimensions 200 \
#     --batch_size 32 \
#     --learning_rate 0.00001 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.3 \
#     --residual distinct_source \
#     --heads 2 \
#     --positional_encoding
# done


# configs=(k562_allloopshicfdr_cosine.yaml k562_allloopshicfdr_global.yaml)
# for config in "${configs[@]}"; do
#   python ogl/omics_graph_learning/ogl_pipeline.py \
#     --partition RM \
#     --experiment_yaml ogl/configs/experiments/"${config}" \
#     --target rna_seq \
#     --model GAT \
#     --gnn_layers 2 \
#     --linear_layers 2 \
#     --activation gelu \
#     --dimensions 200 \
#     --batch_size 32 \
#     --learning_rate 0.00001 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.3 \
#     --residual distinct_source \
#     --heads 2 \
#     --positional_encoding
# done

# configs=(all_celllines_alloopshicfdr.yaml)
# for config in "${configs[@]}"; do
#   python ogl/omics_graph_learning/ogl_pipeline.py \
#     --partition RM \
#     --experiment_yaml ogl/configs/experiments/"${config}" \
#     --target rna_seq \
#     --model PNA \
#     --gnn_layers 2 \
#     --linear_layers 2 \
#     --activation gelu \
#     --dimensions 200 \
#     --batch_size 32 \
#     --learning_rate 0.0005 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.5 \
#     --residual distinct_source \
#     --positional_encoding
# done

# configs=(k562_allloopshicfdr_global.yaml)
# for config in "${configs[@]}"; do
#   python ogl/omics_graph_learning/ogl_pipeline.py \
#     --partition RM \
#     --experiment_yaml ogl/configs/experiments/"${config}" \
#     --target rna_seq \
#     --model PNA \
#     --gnn_layers 2 \
#     --linear_layers 2 \
#     --activation gelu \
#     --dimensions 256 \
#     --batch_size 64 \
#     --learning_rate 0.0005 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.3 \
#     --residual distinct_source \
#     --positional_encoding \
#     --run_number 3
# done

# configs=(k562_allloopshicfdr.yaml k562_allloopshicfdr_global.yaml)
# for config in "${configs[@]}"; do
#   python ogl/omics_graph_learning/ogl_pipeline.py \
#     --partition RM \
#     --experiment_yaml ogl/configs/experiments/"${config}" \
#     --target rna_seq \
#     --model PNA \
#     --gnn_layers 2 \
#     --linear_layers 2 \
#     --activation gelu \
#     --dimensions 256 \
#     --batch_size 64 \
#     --learning_rate 0.0005 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.3 \
#     --residual distinct_source \
#     --positional_encoding
# done

# configs=(k562_allloopshicfdr_cosine.yaml)
# for config in "${configs[@]}"; do
#   python ogl/omics_graph_learning/ogl_pipeline.py \
#     --partition RM \
#     --experiment_yaml ogl/configs/experiments/"${config}" \
#     --target rna_seq \
#     --model GAT \
#     --gnn_layers 2 \
#     --linear_layers 2 \
#     --activation gelu \
#     --dimensions 200 \
#     --batch_size 64 \
#     --learning_rate 0.0005 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.3 \
#     --residual distinct_source \
#     --heads 2 \
#     --positional_encoding
# done