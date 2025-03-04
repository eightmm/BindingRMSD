# BindingRMSD: Protein-Ligand Binding Pose RMSD Prediction

This repository contains code for predicting the Root Mean Square Deviation (RMSD) of protein-ligand binding poses using GNN (Graph Neural Network) models. The models predict both the RMSD value and the probability of the pose's correctness.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/eightmm/BindingRMSD.git
cd BindingRMSD
```

2. Set up a Python environment and install dependencies:
```bash
conda create -n BindingRMSD python=3.11
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
pip install rdkit
pip install meeko (if using (DLG or PDBQT) from AutoDock)
conda activate BindingRMSD
```

## Usage

This repository provides a script to predict the RMSD of protein-ligand binding poses. The prediction is performed using two models: one for RMSD prediction and one for probability estimation.

### Running the Inference

To run the inference and predict RMSD, use the following command:
```bash
python inference.py -r ./example/1KLT.pdb -l ./example/ligands.sdf -o ./result.csv --model_path ./save --device cuda
```
Where:
- `-r` specifies the receptor protein PDB file.
- `-l` specifies the ligand SDF file.
- `-o` specifies the output CSV file for results.
- `--model_path` specifies the directory containing the model weights (`reg.pth` and `bce.pth`).
- `--device` specifies whether to use `cuda` or `cpu`.

### Output

The output will be saved in the specified CSV file and will contain the following columns:
- **Name**: Name or index of the ligand.
- **RMSD**: Predicted RMSD of the ligand pose.
- **Prob**: Predicted probability of the ligand pose being correct.
- **RMSD\*Prob**: Product of RMSD and probability.
- **RMSD+Prob**: Sum of RMSD and probability.

## File Structure

```
.
├── data
│   ├── data.py                # Data loading and preprocessing
│   ├── ligand_atom_feature.py  # Features for ligand atoms
│   ├── protein_atom_feature.py # Features for protein atoms
│   └── utils.py                # Utility functions
├── env.yaml                    # Conda environment setup file
├── example
│   ├── 1KLT.pdb                # Example receptor PDB file
│   ├── ligands.sdf             # Example ligand SDF file
│   └── run.sh                  # Example script to run inference
├── inference.py                # Inference script for RMSD prediction
├── LICENSE                     # License file
├── model
│   ├── GatedGCNLSPE.py         # Model architecture implementation
│   └── model.py                # Prediction model classes
├── README.md                   # This README file
└── save
    ├── bce.pth                 # Saved weights for probability model
    └── reg.pth                 # Saved weights for RMSD model
```

## Example

Below is an example of how to run the code:
```bash
python inference.py     -r ./example/1KLT.pdb     -l ./example/ligands.sdf     -o ./result.csv     --batch_size 128     --model_path ./save     --device cuda
```

The example receptor `1KLT.pdb` and ligand `ligands.sdf` are provided in the `example/` directory. This command will generate a CSV file named `result.csv` containing the predicted RMSD and probability values for each ligand pose.

## Models

The prediction models are based on Gated Graph Neural Networks (GNNs). The models take the protein and ligand graphs as input and output the predicted RMSD and probability for each ligand pose.

- **RMSD Model (`reg.pth`)**: Predicts the RMSD of the ligand pose.
- **Probability Model (`bce.pth`)**: Predicts the probability that the pose is correct.

The model architectures are defined in `model/GatedGCNLSPE.py` and `model/model.py`.

