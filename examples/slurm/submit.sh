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


# BEST PERFORMING
# /ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/models/regulatory_only_k562_allcontacts_global_gat_2layers_200dim_4attnheads
# regulatory_only_k562_allloopshicfdr_global_UniMPTransformer_2layers_200dim
# regulatory_only_imr90_allcontacts_global/run_3/performance.png
# regulatory_only_hepg2_esc_allcontacts_global/run_3/performance.png
# regulatory_only_h1_esc_allcontacts_global/run_3/performance.png
# regulatory_only_gm12878_allcontacts_global/run_3/performance.png


# re-submit
#  26305306           26305308


# train models at different params and with smooth l1 loss
configs=(k562_allcontacts_global.yaml)
for config in "${configs[@]}"; do
  for dimensions in 64 128 200; do
    for heads in 2 4; do
      python ogl/omics_graph_learning/ogl_pipeline.py \
        --partition RM \
        --experiment_yaml ogl/configs/experiments/"${config}" \
        --target rna_seq \
        --model GAT \
        --gnn_layers 2 \
        --linear_layers 2 \
        --activation gelu \
        --dimensions ${dimensions} \
        --batch_size 64 \
        --learning_rate 0.0005 \
        --optimizer AdamW \
        --scheduler cosine \
        --dropout 0.3 \
        --residual distinct_source \
        --heads ${heads} \
        --positional_encoding \
        --model_name regulatory_k562_allcontacts-global_gat_2layers_${dim}dim_${heads}attnheads
    done
  done
done

configs=(k562_allcontacts_global.yaml)
for config in "${configs[@]}"; do
  for dimensions in 64 128 200; do
    for heads in 2 4; do
      python ogl/omics_graph_learning/ogl_pipeline.py \
        --partition RM \
        --experiment_yaml ogl/configs/experiments/"${config}" \
        --target rna_seq \
        --model GAT \
        --gnn_layers 2 \
        --linear_layers 2 \
        --activation gelu \
        --dimensions ${dimensions} \
        --batch_size 64 \
        --learning_rate 0.0005 \
        --optimizer AdamW \
        --scheduler cosine \
        --dropout 0.3 \
        --residual distinct_source \
        --heads ${heads} \
        --positional_encoding \
        --regression_loss_type smooth_l1 \
        --model_name regulatory_k562_allcontacts-global_gat_2layers_${dim}dim_${heads}attnheads_smoothl1
    done
  done
done

# configs=(all_celllines_allcontacts.yaml)
# for config in "${configs[@]}"; do
#   python ogl/omics_graph_learning/ogl_pipeline.py \
#     --partition RM \
#     --experiment_yaml ogl/configs/experiments/"${config}" \
#     --target rna_seq \
#     --model GAT \
#     --gnn_layers 2 \
#     --linear_layers 2 \
#     --activation gelu \
#     --dimensions 128 \
#     --batch_size 64 \
#     --learning_rate 0.0005 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.5 \
#     --residual distinct_source \
#     --heads 2 \
#     --positional_encoding \
#     --model_name regulatory_only_all_celllines_allcontacts_gat_2layers_128dim_2attnheads_highdrop
# done


# configs=(all_celllines_allcontacts.yaml)
# for config in "${configs[@]}"; do
#   python ogl/omics_graph_learning/ogl_pipeline.py \
#     --partition RM \
#     --experiment_yaml ogl/configs/experiments/"${config}" \
#     --target rna_seq \
#     --model GAT \
#     --gnn_layers 2 \
#     --linear_layers 2 \
#     --activation gelu \
#     --dimensions 64 \
#     --batch_size 64 \
#     --learning_rate 0.0005 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.5 \
#     --residual distinct_source \
#     --heads 4 \
#     --positional_encoding \
#     --model_name regulatory_only_all_celllines_allcontacts_gat_2layers_64dim_4attnheads_highdrop
# done

# configs=(all_celllines_allcontacts.yaml)
# for config in "${configs[@]}"; do
#   python ogl/omics_graph_learning/ogl_pipeline.py \
#     --partition RM \
#     --experiment_yaml ogl/configs/experiments/"${config}" \
#     --target rna_seq \
#     --model UniMPTransformer \
#     --gnn_layers 2 \
#     --linear_layers 2 \
#     --activation gelu \
#     --dimensions 64 \
#     --batch_size 64 \
#     --learning_rate 0.0005 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.5 \
#     --residual distinct_source \
#     --heads 4 \
#     --positional_encoding \
#     --model_name regulatory_only_all_celllines_allcontacts_unimp_2layers_64dim_4attnheads_highdrop
# done

# configs=(all_celllines_allcontacts.yaml)
# for config in "${configs[@]}"; do
#   python ogl/omics_graph_learning/ogl_pipeline.py \
#     --partition RM \
#     --experiment_yaml ogl/configs/experiments/"${config}" \
#     --target rna_seq \
#     --model GAT \
#     --gnn_layers 2 \
#     --linear_layers 2 \
#     --activation gelu \
#     --dimensions 64 \
#     --batch_size 64 \
#     --learning_rate 0.0005 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.5 \
#     --residual distinct_source \
#     --heads 2 \
#     --positional_encoding \
#     --attention_task_head \
#     --model_name regulatory_only_all_celllines_allcontacts_gat_2layers_64dim_2attnheads_highdrop_attntask
# done

# configs=(all_celllines_allcontacts.yaml)
# for config in "${configs[@]}"; do
#   python ogl/omics_graph_learning/ogl_pipeline.py \
#     --partition RM \
#     --experiment_yaml ogl/configs/experiments/"${config}" \
#     --target rna_seq \
#     --model GAT \
#     --gnn_layers 2 \
#     --linear_layers 2 \
#     --activation gelu \
#     --dimensions 64 \
#     --batch_size 64 \
#     --learning_rate 0.0001 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.5 \
#     --residual distinct_source \
#     --heads 4 \
#     --positional_encoding \
#     --model_name regulatory_only_all_celllines_allcontacts_gat_2layers_64dim_4attnheads_highdrop_lr1e-4
# done


# for cell in k562 gm12878 hepg2 h1_esc imr90 hmec nhek; do
#   configs=("${cell}_allcontacts_global.yaml")
#   for config in "${configs[@]}"; do
#     python ogl/omics_graph_learning/ogl_pipeline.py \
#       --partition RM \
#       --experiment_yaml ogl/configs/experiments/"${config}" \
#       --target rna_seq \
#       --model GCN \
#       --gnn_layers 6 \
#       --linear_layers 2 \
#       --activation gelu \
#       --dimensions 64 \
#       --batch_size 64 \
#       --learning_rate 0.0005 \
#       --optimizer AdamW \
#       --scheduler cosine \
#       --dropout 0.3 \
#       --residual distinct_source \
#       --positional_encoding \
#       --model_name regulatory_"${cell}"_allcontacts_global_gcn
#   done
# done

# for cell in k562 gm12878 hepg2 h1_esc imr90 hmec nhek; do
#   configs=("${cell}_allcontacts_global.yaml")
#   for config in "${configs[@]}"; do
#     python ogl/omics_graph_learning/ogl_pipeline.py \
#       --partition RM \
#       --experiment_yaml ogl/configs/experiments/"${config}" \
#       --target rna_seq \
#       --model GAT \
#       --gnn_layers 2 \
#       --linear_layers 2 \
#       --activation gelu \
#       --dimensions 128 \
#       --batch_size 64 \
#       --learning_rate 0.0005 \
#       --optimizer AdamW \
#       --scheduler cosine \
#       --dropout 0.3 \
#       --residual distinct_source \
#       --heads 4 \
#       --positional_encoding \
#       --model_name regulatory_"${cell}"_allcontacts_global_gat_2layers_128dim_4attnheads
#   done
# done

# configs=(all_celllines_allcontacts.yaml)
# for config in "${configs[@]}"; do
#   python ogl/omics_graph_learning/ogl_pipeline.py \
#     --partition RM \
#     --experiment_yaml ogl/configs/experiments/"${config}" \
#     --target rna_seq \
#     --model GAT \
#     --gnn_layers 2 \
#     --linear_layers 2 \
#     --activation gelu \
#     --dimensions 128 \
#     --batch_size 64 \
#     --learning_rate 0.0005 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.3 \
#     --residual distinct_source \
#     --heads 4 \
#     --positional_encoding \
#     --model_name regulatory_only_all_celllines_allcontacts_gat_2layers_128dim_4attnheads
# done

# configs=(k562_allcontacts_global.yaml)
# for config in "${configs[@]}"; do
#   python ogl/omics_graph_learning/ogl_pipeline.py \
#     --partition RM \
#     --experiment_yaml ogl/configs/experiments/"${config}" \
#     --target rna_seq \
#     --model GAT \
#     --gnn_layers 2 \
#     --linear_layers 2 \
#     --activation gelu \
#     --dimensions 128 \
#     --batch_size 64 \
#     --learning_rate 0.0005 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.5 \
#     --residual distinct_source \
#     --heads 2 \
#     --positional_encoding \
#     --model_name regulatory_only_k562_allcontacts_global_gat_2layers_128dim_2attnheads_highdrop
# done

# configs=(k562_allcontacts_global.yaml)
# for config in "${configs[@]}"; do
#   python ogl/omics_graph_learning/ogl_pipeline.py \
#     --partition RM \
#     --experiment_yaml ogl/configs/experiments/"${config}" \
#     --target rna_seq \
#     --model UniMPTransformer \
#     --gnn_layers 2 \
#     --linear_layers 2 \
#     --activation gelu \
#     --dimensions 128 \
#     --batch_size 64 \
#     --learning_rate 0.0005 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.3 \
#     --residual distinct_source \
#     --heads 2 \
#     --positional_encoding \
#     --model_name regulatory_only_k562_allcontacts_global_unimp_2layers_128dim_2attnheads
# done

# configs=(k562_allcontacts_global.yaml)
# for config in "${configs[@]}"; do
#   python ogl/omics_graph_learning/ogl_pipeline.py \
#     --partition RM \
#     --experiment_yaml ogl/configs/experiments/"${config}" \
#     --target rna_seq \
#     --model GAT \
#     --gnn_layers 2 \
#     --linear_layers 2 \
#     --activation gelu \
#     --dimensions 128 \
#     --batch_size 64 \
#     --learning_rate 0.0005 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.3 \
#     --residual distinct_source \
#     --heads 2 \
#     --positional_encoding \
#     --attention_task_head \
#     --model_name regulatory_only_k562_allcontacts_global_gat_2layers_128dim_2attnheads_attntask
# done

# configs=(k562_allcontacts_global.yaml)
# for config in "${configs[@]}"; do
#   python ogl/omics_graph_learning/ogl_pipeline.py \
#     --partition RM \
#     --experiment_yaml ogl/configs/experiments/"${config}" \
#     --target rna_seq \
#     --model GAT \
#     --gnn_layers 2 \
#     --linear_layers 2 \
#     --activation gelu \
#     --dimensions 32 \
#     --batch_size 64 \
#     --learning_rate 0.0005 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.3 \
#     --residual distinct_source \
#     --heads 4 \
#     --positional_encoding \
#     --model_name regulatory_only_k562_allcontacts_global_gat_2layers_32dim_4attnheads
# done



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
#     --dimensions 128 \
#     --batch_size 64 \
#     --learning_rate 0.0005 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.5 \
#     --residual distinct_source \
#     --heads 4 \
#     --positional_encoding \
#     --model_name regulatory_only_all_celllines_alloopshicfdr_gat_2layers_200dim_4attnheads_highdrop
# done


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
#     --dimensions 128 \
#     --batch_size 64 \
#     --learning_rate 0.0005 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.5 \
#     --residual distinct_source \
#     --positional_encoding \
#     --attention_task_head \
#     --model_name regulatory_only_all_celllines_alloopshicfdr_gat_2layers_200dim_attntask_highdrop
# done

# configs=(k562_allloopshicfdr_global.yaml)
# for config in "${configs[@]}"; do
#   python ogl/omics_graph_learning/ogl_pipeline.py \
#     --partition RM \
#     --experiment_yaml ogl/configs/experiments/"${config}" \
#     --target rna_seq \
#     --model UniMPTransformer \
#     --gnn_layers 2 \
#     --linear_layers 2 \
#     --activation gelu \
#     --dimensions 128 \
#     --batch_size 64 \
#     --learning_rate 0.0005 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.3 \
#     --residual distinct_source \
#     --heads 2 \
#     --positional_encoding \
#     --model_name regulatory_only_k562_allloopshicfdr_global_UniMPTransformer_2layers_200dim
# done

# configs=(k562_allcontacts_global.yaml)
# for config in "${configs[@]}"; do
#   python ogl/omics_graph_learning/ogl_pipeline.py \
#     --partition RM \
#     --experiment_yaml ogl/configs/experiments/"${config}" \
#     --target rna_seq \
#     --model UniMPTransformer \
#     --gnn_layers 2 \
#     --linear_layers 2 \
#     --activation gelu \
#     --dimensions 128 \
#     --batch_size 64 \
#     --learning_rate 0.0005 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.3 \
#     --residual distinct_source \
#     --heads 2 \
#     --positional_encoding \
#     --model_name regulatory_only_k562_allcontacts_global_UniMPTransformer_2layers_200dim
# done

# configs=(k562_allloopshicfdr_global.yaml)
# for config in "${configs[@]}"; do
#   python ogl/omics_graph_learning/ogl_pipeline.py \
#     --partition RM \
#     --experiment_yaml ogl/configs/experiments/"${config}" \
#     --target rna_seq \
#     --model GAT \
#     --gnn_layers 2 \
#     --linear_layers 2 \
#     --activation gelu \
#     --dimensions 128 \
#     --batch_size 64 \
#     --learning_rate 0.0005 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.3 \
#     --residual distinct_source \
#     --heads 2 \
#     --positional_encoding \
#     --attention_task_head \
#     --model_name regulatory_only_k562_allloopshicfdr_global_gat_2layers_200dim_attntask
# done

# configs=(k562_allcontacts_global.yaml)
# for config in "${configs[@]}"; do
#   python ogl/omics_graph_learning/ogl_pipeline.py \
#     --partition RM \
#     --experiment_yaml ogl/configs/experiments/"${config}" \
#     --target rna_seq \
#     --model GAT \
#     --gnn_layers 2 \
#     --linear_layers 2 \
#     --activation gelu \
#     --dimensions 128 \
#     --batch_size 64 \
#     --learning_rate 0.0005 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.3 \
#     --residual distinct_source \
#     --heads 2 \
#     --positional_encoding \
#     --attention_task_head \
#     --model_name regulatory_only_k562_allcontacts_global_gat_2layers_200dim_attntask
# done

# configs=(k562_allloopshicfdr_global.yaml)
# for config in "${configs[@]}"; do
#   python ogl/omics_graph_learning/ogl_pipeline.py \
#     --partition RM \
#     --experiment_yaml ogl/configs/experiments/"${config}" \
#     --target rna_seq \
#     --model GAT \
#     --gnn_layers 2 \
#     --linear_layers 2 \
#     --activation gelu \
#     --dimensions 128 \
#     --batch_size 64 \
#     --learning_rate 0.0005 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.3 \
#     --residual distinct_source \
#     --heads 4 \
#     --positional_encoding \
#     --model_name regulatory_only_k562_allloopshicfdr_global_gat_2layers_200dim_4attnheads
# done

# configs=(k562_allcontacts_global.yaml)
# for config in "${configs[@]}"; do
#   python ogl/omics_graph_learning/ogl_pipeline.py \
#     --partition RM \
#     --experiment_yaml ogl/configs/experiments/"${config}" \
#     --target rna_seq \
#     --model GAT \
#     --gnn_layers 2 \
#     --linear_layers 2 \
#     --activation gelu \
#     --dimensions 128 \
#     --batch_size 64 \
#     --learning_rate 0.0005 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.3 \
#     --residual distinct_source \
#     --heads 4 \
#     --positional_encoding \
#     --model_name regulatory_only_k562_allcontacts_global_gat_2layers_200dim_4attnheads
# done

# configs=(k562_allloopshicfdr_global.yaml)
# for config in "${configs[@]}"; do
#   python ogl/omics_graph_learning/ogl_pipeline.py \
#     --partition RM \
#     --experiment_yaml ogl/configs/experiments/"${config}" \
#     --target rna_seq \
#     --model GAT \
#     --gnn_layers 2 \
#     --linear_layers 2 \
#     --activation gelu \
#     --dimensions 128 \
#     --batch_size 64 \
#     --learning_rate 0.0005 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.5 \
#     --residual distinct_source \
#     --heads 2 \
#     --positional_encoding \
#     --model_name regulatory_only_k562_allloopshicfdr_global_gat_2layers_200dim_highdrop
# done

# configs=(k562_allcontacts_global.yaml)
# for config in "${configs[@]}"; do
#   python ogl/omics_graph_learning/ogl_pipeline.py \
#     --partition RM \
#     --experiment_yaml ogl/configs/experiments/"${config}" \
#     --target rna_seq \
#     --model GAT \
#     --gnn_layers 2 \
#     --linear_layers 2 \
#     --activation gelu \
#     --dimensions 128 \
#     --batch_size 64 \
#     --learning_rate 0.0005 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.5 \
#     --residual distinct_source \
#     --heads 2 \
#     --positional_encoding \
#     --model_name regulatory_only_k562_allcontacts_global_gat_2layers_200dim_highdrop
# done


# configs=(k562_allloopshicfdr_global.yaml)
# for config in "${configs[@]}"; do
#   python ogl/omics_graph_learning/ogl_pipeline.py \
#     --partition RM \
#     --experiment_yaml ogl/configs/experiments/"${config}" \
#     --target rna_seq \
#     --model GAT \
#     --gnn_layers 3 \
#     --linear_layers 2 \
#     --activation gelu \
#     --dimensions 128 \
#     --batch_size 64 \
#     --learning_rate 0.0005 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.3 \
#     --residual distinct_source \
#     --heads 2 \
#     --positional_encoding \
#     --model_name regulatory_only_k562_allloopshicfdr_global_gat_3layers_128dim
# done

# configs=(k562_allcontacts_global.yaml)
# for config in "${configs[@]}"; do
#   python ogl/omics_graph_learning/ogl_pipeline.py \
#     --partition RM \
#     --experiment_yaml ogl/configs/experiments/"${config}" \
#     --target rna_seq \
#     --model GAT \
#     --gnn_layers 3 \
#     --linear_layers 2 \
#     --activation gelu \
#     --dimensions 128 \
#     --batch_size 64 \
#     --learning_rate 0.0005 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.3 \
#     --residual distinct_source \
#     --heads 2 \
#     --positional_encoding \
#     --model_name regulatory_only_k562_allcontacts_global_gat_3layers_128dim
# done


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
#     --dimensions 128 \
#     --batch_size 64 \
#     --learning_rate 0.0005 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.3 \
#     --residual distinct_source \
#     --heads 2 \
#     --positional_encoding
# done


# configs=(k562_allcontacts_global.yaml)
# for config in "${configs[@]}"; do
#   python ogl/omics_graph_learning/ogl_pipeline.py \
#     --partition RM \
#     --experiment_yaml ogl/configs/experiments/"${config}" \
#     --target rna_seq \
#     --model GAT \
#     --gnn_layers 2 \
#     --linear_layers 2 \
#     --activation gelu \
#     --dimensions 128 \
#     --batch_size 64 \
#     --learning_rate 0.0005 \
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
#     --dimensions 128 \
#     --batch_size 64 \
#     --learning_rate 0.0005 \
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
#     --dimensions 128 \
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
#     --dimensions 128 \
#     --batch_size 64 \
#     --learning_rate 0.0005 \
#     --optimizer AdamW \
#     --scheduler cosine \
#     --dropout 0.3 \
#     --residual distinct_source \
#     --heads 2 \
#     --positional_encoding
# done