# GraDKT

Welcome to the paper **"Gaussian Graph Diffusion with Response Reconstruction for Knowledge Tracing"** .


## Model Architecture

<img alt="GraDKT" src="assets/GraDKT_framework.jpg">


## Experiment Environment
- python 3.10+
- torch 2.0+
- torch_geometric 2.4+
- scikit-learn 1.4+
- pandas 2.2.0+
- tqdm

**Environment Setting Example**

```bash
pip3 install torch torchvision torchaudio
pip3 install torch_geometric
pip3 install scikit-learn pandas tqdm
```


## Dataset Download

To use the GraDKT model, you first need to download and prepare the necessary datasets. The table below provides the download links and the required files for each dataset.

| **Dataset**  | **Download**                                                                                                          | **Required Files**                                                                                                             |
|--------------|-----------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| **DBE-KT22** | <a href="https://dataverse.ada.edu.au/dataset.xhtml?persistentId=doi:10.26193/6DZWOH" target="_blank">link</a>        | `Questions.csv`, `Question_KC_Relationships.csv`, `KCs.csv`, `Question_Choices.csv`, `Transaction.csv`, `KC_Relationships.csv` |
| **EdNet**    | <a href="https://drive.google.com/file/d/1yLNUGvdWCY9tnX4L-MRfp2c2FYQWGKLm/view?usp=sharing" target="_blank">link</a> | `kt_ednet.json`                                                                                                                |
| **NIPS34**   | <a href="https://eedi.com/projects/neurips-education-challenge" target="_blank">link</a>                              | `question_metadata_task_3_4.csv`, `train_task_3_4.csv`, `subject_metadata.csv`                                                 |
| **ENEM**     | <a href="https://github.com/godtn0/DP-MTL/blob/main/data/enem_data/enem_dep.zip" target="_blank">link</a>             | `enem_dep.csv`                                                                                                                 |


### Setup Instructions

1. **Download** the datasets from the links provided in the table above.
2. **Extract** the files into the respective directories under `data/{dataset_name}/raw`. Ensure all required files are available in these directories before proceeding with preprocessing and experimentation.


## Basic Usage


### Preprocessing Datasets

Before running experiments with the GraDKT model, you must preprocess the datasets. Use the following command to preprocess the dataset:

```bash
python3 data_preprocess.py --dataset {dataset_name} \
```

Replace {dataset_name} with the name of the folder containing the dataset you want to preprocess. This should match the folder name within the `data` directory.


### Run Experiment

To run an experiment with the GraDKT model, use the following command:

```bash
python3 main.py --dataset {dataset_name} --model {model} --batch {batch_size} --lr {learning_rate} \
--dim_c {concept_dimension} --dim_q {question_dimension} --dim_g {output_dimension} \
--layer_g {ggn_layers} --lamb {lambda_value} --alpha {top_k_coefficient} --top_k {k_value} \
--beta {STATE_contrastive_loss_coefficient} --gamma {GRAPH_contrastive_loss_coefficient} --recon_weight {Reconstruction_loss_coefficient} --exp_name {experiment_name}
```

**Example Command:**

To run an experiment with the DBE-KT22 dataset and the GraDKT model with a batch size of 128, learning rate of 1e-3, and specified dimensions and coefficients, the command would look like:

```bash
python3 main.py --dataset DBE_KT22 --model GraDKT --batch 128 --lr 1e-3 \
--dim_c 32 --dim_q 32 --dim_g 32 --layer_g 3 --lamb 0.1 --top_k 15 \
--alpha 0.01 --beta 0.01 --gamma 0.01 --recon_weight 0.01 --exp_name "1"
```
