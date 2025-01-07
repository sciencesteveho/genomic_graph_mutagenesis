
# Omics Graph Learning CLI Arguments

Below is a detailed description of the available command_line arguments used by the `OGLCLIParser` class.

## Table of Contents

- [Base Arguments](#base-arguments)
- [Model Arguments](#model-arguments)
- [Boolean Flags](#boolean-flags)
- [Perturbation Arguments](#perturbation-arguments)
- [Specific GNN Training Arguments](#specific-gnn-training-arguments)
- [Validation Rules](#validation-rules)

---

## Base Arguments

These arguments are essential for configuring the overall experiment and model setup.

| Argument                    | Type    | Default                 | Choices                                       | Description                                                                                                                          |
|-----------------------------|---------|-------------------------|-----------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `--experiment_yaml`         | `str`   | _Required_              | N/A                                           | Path to the experiment YAML file.                                                                                                   |
| `--partition`               | `str`   | `"RM"`                  | `"RM"`, `"EM"`                                 | Partition for SLURM scheduling.                                                                                                     |
| `--tpm_filter`              | `float` | `0.5`                   | N/A                                           | TPM (Transcripts Per Million) filter threshold.                                                                                      |
| `--percent_of_samples_filter` | `float` | `0.1`                   | N/A                                           | Percentage of samples to filter.                                                                                                     |
| `--filter_mode`             | `str`   | `"within"`              | `"within"`, `"across"`                         | Mode to filter genes, specifying within the target tissue or across all possible GTEx tissues. Required if the target type is not `rna_seq`. |
| `--clean-up`                | Flag    | `False`                 | N/A                                           | Remove intermediate files in tissue-specific directories.                                                                            |
| `--n_gpus`                  | `int`   | _None_                  | N/A                                           | Number of GPUs to use.                                                                                                               |
| `--model_name`              | `str`   | `None`                  | N/A                                           | Alternative model name.                                                                                                              |

---

## Model Arguments

These arguments pertain to the configuration and training of the Graph Neural Network (GNN) model.

| Argument                    | Type    | Default                 | Choices                                       | Description                                                                                                                          |
|-----------------------------|---------|-------------------------|-----------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `--model`                   | `str`   | `"GCN"`                 | `"GCN"`, `"GraphSAGE"`, `"PNA"`, `"GAT"`, `"UniMPTransformer"`, `"DeeperGCN"`, `"MLP"` | Type of GNN model to use.                                                                                                             |
| `--target`                  | `str`   | `"expression_median_only"` | `"expression_median_only"`, `"expression_media_and_foldchange"`, `"difference_from_average"`, `"foldchange_from_average"`, `"protein_targets"`, `"rna_seq"` | Target prediction type.                                                                                                              |
| `--gnn_layers`              | `int`   | `2`                     | N/A                                           | Number of GNN layers.                                                                                                                |
| `--linear_layers`           | `int`   | `3`                     | N/A                                           | Number of linear layers.                                                                                                             |
| `--activation`              | `str`   | `"relu"`                | `"relu"`, `"leakyrelu"`, `"gelu"`              | Activation function to use.                                                                                                         |
| `--dimensions`              | `int`   | `256`                   | N/A                                           | Dimension size for layers.                                                                                                           |
| `--residual`                | `str`   | `None`                  | `"shared_source"`, `"distinct_source"`, `"None"` | Type of residual connection.                                                                                                         |
| `--epochs`                  | `int`   | `60`                    | N/A                                           | Number of training epochs.                                                                                                           |
| `--batch_size`              | `int`   | `256`                   | N/A                                           | Batch size for training.                                                                                                             |
| `--learning_rate`           | `float` | `1e-4`                  | N/A                                           | Learning rate for the optimizer.                                                                                                     |
| `--optimizer`               | `str`   | `"Adam"`                | `"Adam"`, `"AdamW"`                             | Optimizer to use for training.                                                                                                       |
| `--scheduler`               | `str`   | `"plateau"`             | `"plateau"`, `"cosine"`, `"linear_warmup"`      | Learning rate scheduler type.                                                                                                        |
| `--regression_loss_type`    | `str`   | `"rmse"`                | `"rmse"`, `"smooth_l1"`                         | Type of regression loss function.                                                                                                    |
| `--dropout`                 | `float` | `0.1`                   | N/A                                           | Dropout rate.                                                                                                                        |
| `--heads`                   | `int`   | `None`                  | N/A                                           | Number of attention heads (required for certain models).                                                                             |
| `--n_trials`                | `int`   | _None_                  | N/A                                           | Number of trials for hyperparameter optimization.                                                                                    |
| `--run_number`              | `int`   | _None_                  | N/A                                           | Run number to specify for GNN training. If not specified, the pipeline submits three jobs [0, 1, 2] and trains three models across three seeds. |

---

## Boolean Flags

These flags toggle specific features or behaviors in the pipeline.

| Argument               | Default | Description                                                                                                      |
|------------------------|---------|------------------------------------------------------------------------------------------------------------------|
| `--attention_task_head` | `False` | Enable attention task head.                                                                                      |
| `--positional_encoding` | `False` | Enable positional encoding.                                                                                       |
| `--early_stop`          | `True`  | Enable early stopping.                                                                                             |
| `--gene_only_loader`    | `False` | Use gene-only subgraph loader.                                                                                         |
| `--optimize_params`     | `False` | Enable hyperparameter optimization.                                                                               |


---

## Specific GNN Training Arguments

These arguments are specifically related to the training process of the GNN model.

| Argument   | Type  | Default | Description                                                                                          |
|------------|-------|---------|------------------------------------------------------------------------------------------------------|
| `--split_name` | `str` | _Required_ | Name of the data split to use for training (e.g., train, validation, test).                        |
| `--seed`       | `int` | `42`    | Random seed to use for reproducibility.                                                              |
| `--device`     | `int` | `0`     | GPU device index to use for training (default is GPU 0). If not using GPUs, set appropriately.       |

---

## Validation Rules

The parser includes several validation rules to ensure the consistency and correctness of the provided arguments:

1. **Filter Mode Requirement**:
   - If `--target` is not `"rna_seq"`, the `--filter_mode` argument **must** be specified.

<br>

2. **Heads Requirement for Certain Models**:
   - If `--model` is set to `"GAT"` or `"UniMPTransformer"`, the `--heads` argument **must** be specified.

<br>

3. **GPU Requirement for Parameter Optimization**:
   - If `--optimize_params` is enabled, the `--n_gpus` argument **must** be specified.
   
<br>

4. **Edge Perturbation Consistency**:
   - If `--total_random_edges` is set, `--edge_perturbation` **must** be `"randomize_edges"`.
   
<br>

Ensure that these conditions are met when executing the pipeline to avoid runtime errors.

