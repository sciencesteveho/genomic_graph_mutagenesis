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


# Best release model so far
# k562_release_best_params_GAT_dropout_0.1_smoothl1 

# submit best model + randomized node features per idx
for node in {5..41}; do
  python ogl/omics_graph_learning/ogl_pipeline.py \
    --partition RM \
    --experiment_yaml ogl/configs/experiments/k562_release.yaml \
    --target rna_seq \
    --model GAT \
    --gnn_layers 2 \
    --linear_layers 2 \
    --activation gelu \
    --dimensions 200 \
    --batch_size 16 \
    --learning_rate 0.0005 \
    --optimizer AdamW \
    --scheduler cosine \
    --dropout 0.1 \
    --residual distinct_source \
    --heads 2 \
    --positional_encoding \
    --alpha 0.95 \
    --regression_loss_type smooth_l1 \
    --randomize_node_feature_idx ${node} \
    --model_name k562_release_randomize_node_${node}
done

# k562_release_all_nodes_and_interact.yaml
python ogl/omics_graph_learning/ogl_pipeline.py \
  --partition RM \
  --experiment_yaml ogl/configs/experiments/k562_release_all_nodes_and_interact.yaml \
  --target rna_seq \
  --model GAT \
  --gnn_layers 2 \
  --linear_layers 2 \
  --activation gelu \
  --dimensions 200 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --optimizer AdamW \
  --scheduler cosine \
  --dropout 0.1 \
  --residual distinct_source \
  --heads 2 \
  --positional_encoding \
  --regression_loss_type smooth_l1 \
  --model_name k562_release_all_nodes_and_interact

python ogl/omics_graph_learning/ogl_pipeline.py \
  --partition RM \
  --experiment_yaml ogl/configs/experiments/k562_release_all_interact.yaml \
  --target rna_seq \
  --model GAT \
  --gnn_layers 2 \
  --linear_layers 2 \
  --activation gelu \
  --dimensions 200 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --optimizer AdamW \
  --scheduler cosine \
  --dropout 0.1 \
  --residual distinct_source \
  --heads 2 \
  --positional_encoding \
  --regression_loss_type smooth_l1 \
  --model_name k562_release_all_interact

# submit interaction types, mirna and rbp
configs=(mirna rbp)
for config in "${configs[@]}"; do
  python ogl/omics_graph_learning/ogl_pipeline.py \
    --partition RM \
    --experiment_yaml ogl/configs/experiments/k562_allcontacts_global_"${config}".yaml \
    --target rna_seq \
    --model GAT \
    --gnn_layers 2 \
    --linear_layers 2 \
    --activation gelu \
    --dimensions 200 \
    --batch_size 16 \
    --learning_rate 0.0005 \
    --optimizer AdamW \
    --scheduler cosine \
    --dropout 0.1 \
    --residual distinct_source \
    --heads 2 \
    --positional_encoding \
    --regression_loss_type smooth_l1 \
    --model_name k562_release_"${config}"
done

### Notes
# potentially beneficial node types
# ctcf, cpgislands, tfbindingsites
# all node types
# ctcf, cpgislands, tss, tfbindingsites, crms, se

configs=(all_nodes)
for config in "${configs[@]}"; do
  python ogl/omics_graph_learning/ogl_pipeline.py \
    --partition RM \
    --experiment_yaml ogl/configs/experiments/k562_allcontacts_"${config}".yaml \
    --target rna_seq \
    --model GAT \
    --gnn_layers 2 \
    --linear_layers 2 \
    --activation gelu \
    --dimensions 200 \
    --batch_size 16 \
    --learning_rate 0.0005 \
    --optimizer AdamW \
    --scheduler cosine \
    --dropout 0.1 \
    --residual distinct_source \
    --heads 2 \
    --positional_encoding \
    --regression_loss_type smooth_l1 \
    --model_name k562_release_"${config}"
done

# replicate best model
for rep in 1 2 3; do
  python ogl/omics_graph_learning/ogl_pipeline.py \
    --partition RM \
    --experiment_yaml ogl/configs/experiments/k562_release.yaml \
    --target rna_seq \
    --model GAT \
    --gnn_layers 2 \
    --linear_layers 2 \
    --activation gelu \
    --dimensions 200 \
    --batch_size 16 \
    --learning_rate 0.0005 \
    --optimizer AdamW \
    --scheduler cosine \
    --dropout 0.1 \
    --residual distinct_source \
    --heads 2 \
    --positional_encoding \
    --regression_loss_type smooth_l1 \
    --model_name k562_release_replicate_"${rep}"
done

# submit model with adding different node types
# cpgislands
# crms
# ctcfccre
# superenahcners (check that these are in graph construction too)
# tfbindingsites
# tss annotations 
configs=(cpgislands crms ctcf superenhancers tfbindingsites tss)
for config in "${configs[@]}"; do
  python ogl/omics_graph_learning/ogl_pipeline.py \
    --partition RM \
    --experiment_yaml ogl/configs/experiments/k562_allcontacts_"${config}".yaml \
    --target rna_seq \
    --model GAT \
    --gnn_layers 2 \
    --linear_layers 2 \
    --activation gelu \
    --dimensions 200 \
    --batch_size 16 \
    --learning_rate 0.0005 \
    --optimizer AdamW \
    --scheduler cosine \
    --dropout 0.1 \
    --residual distinct_source \
    --heads 2 \
    --positional_encoding \
    --regression_loss_type smooth_l1 \
    --model_name k562_release_"${config}"
done

# submit model with base-graph only
# --graph_type base
configs=(k562_allcontacts_global_localonly.yaml)
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
    --batch_size 16 \
    --learning_rate 0.0005 \
    --optimizer AdamW \
    --scheduler cosine \
    --dropout 0.1 \
    --residual distinct_source \
    --heads 2 \
    --positional_encoding \
    --regression_loss_type smooth_l1 \
    --model_name k562_release_localonly
done

# submit model with gene-gene interactions
# gene-gene in config
configs=(k562_allcontacts_global_gene_gene.yaml)
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
    --batch_size 16 \
    --learning_rate 0.0005 \
    --optimizer AdamW \
    --scheduler cosine \
    --dropout 0.1 \
    --residual distinct_source \
    --heads 2 \
    --positional_encoding \
    --regression_loss_type smooth_l1 \
    --model_name k562_release_gene_gene
done

# submit models with regulatory schemas
# encode
# epimap
# k562_allcontacts_global_epimap
configs=(epimap encode)
for config in "${configs[@]}"; do
  python ogl/omics_graph_learning/ogl_pipeline.py \
    --partition RM \
    --experiment_yaml ogl/configs/experiments/k562_allcontacts_global_"${config}".yaml \
    --target rna_seq \
    --model GAT \
    --gnn_layers 2 \
    --linear_layers 2 \
    --activation gelu \
    --dimensions 200 \
    --batch_size 16 \
    --learning_rate 0.0005 \
    --optimizer AdamW \
    --scheduler cosine \
    --dropout 0.1 \
    --residual distinct_source \
    --heads 2 \
    --positional_encoding \
    --regression_loss_type smooth_l1 \
    --model_name k562_release_"${config}"
done





for model in GAT; do
  for dropout in 0.1; do
    python ogl/omics_graph_learning/ogl_pipeline.py \
      --partition RM \
      --experiment_yaml ogl/configs/experiments/k562_release.yaml \
      --target rna_seq \
      --model ${model} \
      --gnn_layers 2 \
      --linear_layers 2 \
      --activation gelu \
      --dimensions 200 \
      --batch_size 8 \
      --learning_rate 0.0005 \
      --optimizer AdamW \
      --scheduler cosine \
      --dropout ${dropout} \
      --residual distinct_source \
      --heads 2 \
      --positional_encoding \
      --alpha 0.95 \
      --regression_loss_type smooth_l1 \
      --model_name k562_release_best_params_"${model}"_dropout_"${dropout}"_batch_8
  done
done




for model in GAT; do
  for dropout in 0.1 0.3; do
    python ogl/omics_graph_learning/ogl_pipeline.py \
      --partition RM \
      --experiment_yaml ogl/configs/experiments/k562_release.yaml \
      --target rna_seq \
      --model ${model} \
      --gnn_layers 2 \
      --linear_layers 2 \
      --activation gelu \
      --dimensions 200 \
      --batch_size 16 \
      --learning_rate 0.0005 \
      --optimizer AdamW \
      --scheduler cosine \
      --dropout ${dropout} \
      --residual distinct_source \
      --heads 2 \
      --positional_encoding \
      --alpha 0.95 \
      --regression_loss_type smooth_l1 \
      --model_name k562_release_best_params_"${model}"_dropout_"${dropout}"_batch_16
  done
done

for model in UniMPTransformer; do
  for dropout in 0.1 0.3; do
    python ogl/omics_graph_learning/ogl_pipeline.py \
      --partition RM \
      --experiment_yaml ogl/configs/experiments/k562_release.yaml \
      --target rna_seq \
      --model ${model} \
      --gnn_layers 2 \
      --linear_layers 2 \
      --activation gelu \
      --dimensions 200 \
      --batch_size 16 \
      --learning_rate 0.0005 \
      --optimizer AdamW \
      --scheduler cosine \
      --dropout ${dropout} \
      --residual distinct_source \
      --heads 2 \
      --positional_encoding \
      --alpha 0.95 \
      --regression_loss_type smooth_l1 \
      --model_name k562_release_best_params_"${model}"_dropout_"${dropout}"_batch_16
  done
done

# best so far! release versions
# alpha 0.95
# batch 32
# dropout default of 0.3 and 0.1
# lr default of 0.0005
# best models so far: GAT, PNA, and UniMPTransformer
# new best release model
for model in GAT PNA UniMPTransformer; do
  for dropout in 0.3 0.1; do
    python ogl/omics_graph_learning/ogl_pipeline.py \
      --partition RM \
      --experiment_yaml ogl/configs/experiments/k562_release.yaml \
      --target rna_seq \
      --model ${model} \
      --gnn_layers 2 \
      --linear_layers 2 \
      --activation gelu \
      --dimensions 200 \
      --batch_size 32 \
      --learning_rate 0.0005 \
      --optimizer AdamW \
      --scheduler cosine \
      --dropout ${dropout} \
      --residual distinct_source \
      --heads 2 \
      --positional_encoding \
      --alpha 0.95 \
      --model_name k562_release_best_params_"${model}"_dropout_"${dropout}"
  done
done

# and with smooth l1 loss
for model in GAT PNA UniMPTransformer; do
  for dropout in 0.3 0.1; do
    python ogl/omics_graph_learning/ogl_pipeline.py \
      --partition RM \
      --experiment_yaml ogl/configs/experiments/k562_release.yaml \
      --target rna_seq \
      --model ${model} \
      --gnn_layers 2 \
      --linear_layers 2 \
      --activation gelu \
      --dimensions 200 \
      --batch_size 32 \
      --learning_rate 0.0005 \
      --optimizer AdamW \
      --scheduler cosine \
      --dropout ${dropout} \
      --residual distinct_source \
      --heads 2 \
      --positional_encoding \
      --alpha 0.95 \
      --regression_loss_type smooth_l1 \
      --model_name k562_release_best_params_"${model}"_dropout_"${dropout}"_smoothl1
  done
done


# adjust models
for model in GAT GCN GraphSAGE PNA UniMPTransformer; do
  python ogl/omics_graph_learning/ogl_pipeline.py \
    --partition RM \
    --experiment_yaml ogl/configs/experiments/k562_release.yaml \
    --target rna_seq \
    --model ${model} \
    --gnn_layers 2 \
    --linear_layers 2 \
    --activation gelu \
    --dimensions 200 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --optimizer AdamW \
    --scheduler cosine \
    --dropout 0.3 \
    --residual distinct_source \
    --heads 2 \
    --positional_encoding \
    --alpha 0.95 \
    --model_name k562_release_alpha_0.95_"${model}"
done

# adjust batch size
for batchsize in 32 128 256; do
  python ogl/omics_graph_learning/ogl_pipeline.py \
    --partition RM \
    --experiment_yaml ogl/configs/experiments/k562_release.yaml \
    --target rna_seq \
    --model GAT \
    --gnn_layers 2 \
    --linear_layers 2 \
    --activation gelu \
    --dimensions 200 \
    --batch_size ${batchsize} \
    --learning_rate 0.0005 \
    --optimizer AdamW \
    --scheduler cosine \
    --dropout 0.3 \
    --residual distinct_source \
    --heads 2 \
    --positional_encoding \
    --alpha 0.95 \
    --model_name k562_release_alpha_0.95_batch_"${batchsize}"
done

# adjust dropout
# for dropout in 0.4 0.5; do
for dropout in 0.2 0.1; do
  python ogl/omics_graph_learning/ogl_pipeline.py \
    --partition RM \
    --experiment_yaml ogl/configs/experiments/k562_release.yaml \
    --target rna_seq \
    --model GAT \
    --gnn_layers 2 \
    --linear_layers 2 \
    --activation gelu \
    --dimensions 200 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --optimizer AdamW \
    --scheduler cosine \
    --dropout ${dropout} \
    --residual distinct_source \
    --heads 2 \
    --positional_encoding \
    --alpha 0.95 \
    --model_name k562_release_alpha_0.95_dropout_"${dropout}"
done

# adjust learning rate
# for lr in 0.0003 0.0001; do
for lr in 0.001 0.005; do
  python ogl/omics_graph_learning/ogl_pipeline.py \
    --partition RM \
    --experiment_yaml ogl/configs/experiments/k562_release.yaml \
    --target rna_seq \
    --model GAT \
    --gnn_layers 2 \
    --linear_layers 2 \
    --activation gelu \
    --dimensions 200 \
    --batch_size 32 \
    --learning_rate ${lr} \
    --optimizer AdamW \
    --scheduler cosine \
    --dropout 0.3 \
    --residual distinct_source \
    --heads 2 \
    --positional_encoding \
    --alpha 0.95 \
    --model_name k562_release_alpha_0.95_lr_"${lr}"_batch_32
done


#submit differnet alpha
for alpha in 0.65 0.70 0.75 0.80 0.85 0.90 0.95; do
  python ogl/omics_graph_learning/ogl_pipeline.py \
    --partition RM \
    --experiment_yaml ogl/configs/experiments/k562_release.yaml \
    --target rna_seq \
    --model GAT \
    --gnn_layers 2 \
    --linear_layers 2 \
    --activation gelu \
    --dimensions 200 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --optimizer AdamW \
    --scheduler cosine \
    --dropout 0.3 \
    --residual distinct_source \
    --heads 2 \
    --positional_encoding \
    --alpha ${alpha} \
    --model_name k562_release_"${alpha}"
done



# best so far
# regulatory_k562_allcontacts-global_gat_2layers_dim_2attnheads, as well as _replicate and _replicate_2
for rep in 0 1 2; do
  python ogl/omics_graph_learning/ogl_pipeline.py \
    --partition RM \
    --experiment_yaml ogl/configs/experiments/k562_release.yaml \
    --target rna_seq \
    --model GAT \
    --gnn_layers 2 \
    --linear_layers 2 \
    --activation gelu \
    --dimensions 200 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --optimizer AdamW \
    --scheduler cosine \
    --dropout 0.3 \
    --residual distinct_source \
    --heads 2 \
    --positional_encoding \
    --model_name K562_release_rep_"${rep}"
done

cd /ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/experiments && rm -r k562_allcontacts_global_mirna k562_allcontacts_global_rbp && cd - 

# submit interaction types, mirna and rbp
configs=(mirna rbp)
for config in "${configs[@]}"; do
  python ogl/omics_graph_learning/ogl_pipeline.py \
    --partition RM \
    --experiment_yaml ogl/configs/experiments/k562_allcontacts_global_"${config}".yaml \
    --target rna_seq \
    --model GAT \
    --gnn_layers 2 \
    --linear_layers 2 \
    --activation gelu \
    --dimensions 200 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --optimizer AdamW \
    --scheduler cosine \
    --dropout 0.3 \
    --residual distinct_source \
    --heads 2 \
    --positional_encoding \
    --model_name k562_allcontacts_global_"${config}"
done


### Notes
# potentially beneficial node types
# ctcf, cpgislands, tfbindingsites
# all node types
# ctcf, cpgislands, tss, tfbindingsites, crms, se

configs=(add_nodes all_nodes)
for config in "${configs[@]}"; do
  python ogl/omics_graph_learning/ogl_pipeline.py \
    --partition RM \
    --experiment_yaml ogl/configs/experiments/k562_allcontacts_"${config}".yaml \
    --target rna_seq \
    --model GAT \
    --gnn_layers 2 \
    --linear_layers 2 \
    --activation gelu \
    --dimensions 200 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --optimizer AdamW \
    --scheduler cosine \
    --dropout 0.3 \
    --residual distinct_source \
    --heads 2 \
    --positional_encoding \
    --model_name k562_allcontacts_global_"${config}"
done

# replicate best model
python ogl/omics_graph_learning/ogl_pipeline.py \
  --partition RM \
  --experiment_yaml ogl/configs/experiments/k562_allcontacts_global.yaml \
  --target rna_seq \
  --model GAT \
  --gnn_layers 2 \
  --linear_layers 2 \
  --activation gelu \
  --dimensions 200 \
  --batch_size 64 \
  --learning_rate 0.0005 \
  --optimizer AdamW \
  --scheduler cosine \
  --dropout 0.3 \
  --residual distinct_source \
  --heads 2 \
  --positional_encoding \
  --model_name k562_allcontacts_global_replicate_2

# submit model with adding different node types
# cpgislands
# crms
# ctcfccre
# superenahcners (check that these are in graph construction too)
# tfbindingsites
# tss annotations 
configs=(cpgislands crms ctcf superenhancers tfbindingsites tss)
for config in "${configs[@]}"; do
  python ogl/omics_graph_learning/ogl_pipeline.py \
    --partition RM \
    --experiment_yaml ogl/configs/experiments/k562_allcontacts_"${config}".yaml \
    --target rna_seq \
    --model GAT \
    --gnn_layers 2 \
    --linear_layers 2 \
    --activation gelu \
    --dimensions 200 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --optimizer AdamW \
    --scheduler cosine \
    --dropout 0.3 \
    --residual distinct_source \
    --heads 2 \
    --positional_encoding \
    --model_name k562_allcontacts_global_"${config}"
done

# submit model with base-graph only
# --graph_type base
configs=(k562_allcontacts_global_localonly.yaml)
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
    --batch_size 64 \
    --learning_rate 0.0005 \
    --optimizer AdamW \
    --scheduler cosine \
    --dropout 0.3 \
    --residual distinct_source \
    --heads 2 \
    --positional_encoding \
    --model_name k562_allcontacts_global_localonly
done

# submit model with gene-gene interactions
# gene-gene in config
configs=(k562_allcontacts_global_localonly.yaml)
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
    --batch_size 64 \
    --learning_rate 0.0005 \
    --optimizer AdamW \
    --scheduler cosine \
    --dropout 0.3 \
    --residual distinct_source \
    --heads 2 \
    --positional_encoding \
    --model_name k562_allcontacts_global_gene_gene
done

# submit models with regulatory schemas
# encode
# epimap
# k562_allcontacts_global_epimap
configs=(epimap encode)
for config in "${configs[@]}"; do
  python ogl/omics_graph_learning/ogl_pipeline.py \
    --partition RM \
    --experiment_yaml ogl/configs/experiments/k562_allcontacts_global_"${config}".yaml \
    --target rna_seq \
    --model GAT \
    --gnn_layers 2 \
    --linear_layers 2 \
    --activation gelu \
    --dimensions 200 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --optimizer AdamW \
    --scheduler cosine \
    --dropout 0.3 \
    --residual distinct_source \
    --heads 2 \
    --positional_encoding \
    --model_name k562_allcontacts_global_"${config}"
done




# re-run all non k562 models, and try running new tissue models
# for sample in gm12878 hepg2 h1_esc imr90 hmec nhek aorta hippocampus left_ventricle liver lung mammary pancreas skeletal_muscle skin small_intestine; do
# for sample in aorta hippocampus left_ventricle liver lung mammary pancreas skeletal_muscle skin small_intestine; do
for sample in adrenal ovary spleen; do
  config=("${sample}_allcontacts_global.yaml")
  python ogl/omics_graph_learning/ogl_pipeline.py \
    --partition RM \
    --experiment_yaml ogl/configs/experiments/"${config}" \
    --target rna_seq \
    --model GAT \
    --gnn_layers 2 \
    --linear_layers 2 \
    --activation gelu \
    --dimensions 200 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --optimizer AdamW \
    --scheduler cosine \
    --dropout 0.3 \
    --residual distinct_source \
    --heads 2 \
    --positional_encoding \
    --model_name regulatory_${sample}_allcontacts-global_gat_2layers_200dim_2attnheads
done

# train models at different params and with smooth l1 loss
# configs=(k562_allcontacts_global.yaml)
# for config in "${configs[@]}"; do
#   for dimensions in 64 128 200; do
#     for heads in 2 4; do
#       python ogl/omics_graph_learning/ogl_pipeline.py \
#         --partition RM \
#         --experiment_yaml ogl/configs/experiments/"${config}" \
#         --target rna_seq \
#         --model GAT \
#         --gnn_layers 2 \
#         --linear_layers 2 \
#         --activation gelu \
#         --dimensions ${dimensions} \
#         --batch_size 64 \
#         --learning_rate 0.0005 \
#         --optimizer AdamW \
#         --scheduler cosine \
#         --dropout 0.3 \
#         --residual distinct_source \
#         --heads ${heads} \
#         --positional_encoding \
#         --model_name regulatory_k562_allcontacts-global_gat_2layers_${dim}dim_${heads}attnheads
#     done
#   done
# done

# configs=(k562_allcontacts_global.yaml)
# for config in "${configs[@]}"; do
#   for dimensions in 64 128 200; do
#     for heads in 2 4; do
#       python ogl/omics_graph_learning/ogl_pipeline.py \
#         --partition RM \
#         --experiment_yaml ogl/configs/experiments/"${config}" \
#         --target rna_seq \
#         --model GAT \
#         --gnn_layers 2 \
#         --linear_layers 2 \
#         --activation gelu \
#         --dimensions ${dimensions} \
#         --batch_size 64 \
#         --learning_rate 0.0005 \
#         --optimizer AdamW \
#         --scheduler cosine \
#         --dropout 0.3 \
#         --residual distinct_source \
#         --heads ${heads} \
#         --positional_encoding \
#         --regression_loss_type smooth_l1 \
#         --model_name regulatory_k562_allcontacts-global_gat_2layers_${dim}dim_${heads}attnheads_smoothl1
#     done
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