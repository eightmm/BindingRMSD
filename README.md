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
conda activate BindingRMSD
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
pip install rdkit
pip install meeko  # Required if using DLG or PDBQT files from AutoDock
```

## Usage

This repository provides a script to predict the RMSD of protein-ligand binding poses. The prediction is performed using two models: one for RMSD prediction and one for probability estimation.

### Running the Inference

To run the inference and predict RMSD, use the following command:
```bash
python inference.py -r ./example/1KLT.pdb -l ./example/ligands.sdf -o ./result.csv --model_path ./save --device cuda
```
Where:
- `-r`: Receptor protein PDB file
- `-l`: Ligand file (supported formats: .sdf, .mol2, .dlg, .pdbqt)
- `-o`: Output CSV file for results
- `--model_path`: Directory containing model weights (`reg.pth` and `bce.pth`)
- `--device`: Specify `cuda` or `cpu`

### Output

The output will be saved in the specified CSV file with the following columns:
- **Name**: Name or identifier of the ligand pose
- **pRMSD**: Predicted RMSD value for the ligand pose
- **Is_Above_2A**: Predicted probability of the pose being correct (between 0 and 1)
- **ADG_Score**: AutoDock Score (available for .dlg and .pdbqt files, NaN for other formats)

### Input File Formats

The ligand input file supports the following formats:
- `.sdf`: Standard Structure Data File
- `.mol2`: MOL2 file format
- `.dlg`: AutoDock-GPU docking result DLG file
- `.pdbqt`: AutoDock Vina result PDBQT file
- `.txt`: Text file containing a list of paths to any of the above formats

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

