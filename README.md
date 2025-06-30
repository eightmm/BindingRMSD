# 🧬 BindingRMSD

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-red.svg)
![DGL](https://img.shields.io/badge/DGL-2.4.0-green.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)

> **Advanced protein-ligand binding pose RMSD prediction using Graph Neural Networks**

BindingRMSD is a tool for predicting the Root Mean Square Deviation (RMSD) of protein-ligand binding poses using Graph Neural Networks (GNNs). The model provides both RMSD values and confidence scores for binding pose evaluation.

## ✨ Features

- 🎯 **Accurate RMSD Prediction**: State-of-the-art GNN models for precise RMSD estimation
- 📊 **Confidence Scoring**: Probability estimation for pose correctness assessment  
- 🔧 **Multiple Input Formats**: Support for SDF, MOL2, DLG, PDBQT, and batch processing
- ⚡ **GPU Acceleration**: CUDA support for high-performance inference
- 📦 **Easy Installation**: Simple pip-based installation with conda environment
- 🔄 **Batch Processing**: Efficient processing of multiple ligand poses

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/eightmm/BindingRMSD.git
cd BindingRMSD

# Create and activate conda environment
conda create -n bindingrmsd python=3.11
conda activate bindingrmsd

# Install the package
pip install -e .

# Or install dependencies manually
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
pip install torch rdkit meeko pandas tqdm
```

### Basic Usage

#### Command Line Interface

```bash
# Using the installed command
bindingrmsd-inference \
    -r example/prot.pdb \
    -l example/ligs.sdf \
    -o results.tsv \
    --model_path save \
    --device cuda

# Or using the module directly
python -m bindingrmsd.inference \
    -r example/prot.pdb \
    -l example/ligs.sdf \
    -o results.tsv \
    --model_path save \
    --device cuda
```

#### Python API

```python
from bindingrmsd.inference import inference

# Run inference
inference(
    protein_pdb="example/prot.pdb",
    ligand_file="example/ligs.sdf", 
    output="results.tsv",
    batch_size=128,
    model_path="save",
    device="cuda"
)
```

## 📖 Usage Guide

### Input Parameters

| Parameter | Description | Default | Required |
|-----------|-------------|---------|----------|
| `-r, --protein_pdb` | Receptor protein PDB file | - | ✅ |
| `-l, --ligand_file` | Ligand file or file list | - | ✅ |
| `-o, --output` | Output results file | `result.csv` | ❌ |
| `--model_path` | Directory with model weights | `./save` | ❌ |
| `--batch_size` | Batch size for inference | `128` | ❌ |
| `--device` | Compute device (`cuda`/`cpu`) | `cuda` | ❌ |
| `--ncpu` | Number of CPU workers | `4` | ❌ |

### Supported Input Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| SDF | `.sdf` | Structure Data File |
| MOL2 | `.mol2` | Tripos MOL2 format |
| DLG | `.dlg` | AutoDock-GPU results |
| PDBQT | `.pdbqt` | AutoDock Vina results |
| List | `.txt` | Text file with file paths |

### Output Format

The results are saved as a tab-separated file with the following columns:

- **Name**: Ligand pose identifier
- **pRMSD**: Predicted RMSD value (Å)
- **Is_Above_2A**: Confidence score (0-1, probability of being a good pose, 0 is better)
- **ADG_Score**: AutoDock score (when available, NaN otherwise, 0 is better)

## 🏗️ Architecture

### Model Components

- **🧠 Gated Graph Neural Network**: Advanced GNN architecture for molecular representation
- **🔗 Protein-Ligand Interaction**: Comprehensive modeling of binding interactions
- **🎯 Dual Prediction**: Simultaneous RMSD and confidence prediction
- **⚡ Efficient Processing**: Optimized for batch inference

### File Structure

```
BindingRMSD/
├── 📁 bindingrmsd/          # Main package
│   ├── 📁 data/             # Data processing modules
│   │   ├── data.py          # Dataset classes
│   │   ├── ligand_atom_feature.py   # Ligand featurization
│   │   ├── protein_atom_feature.py  # Protein featurization
│   │   └── utils.py         # Utility functions
│   ├── 📁 model/            # Model architecture
│   │   ├── GatedGCNLSPE.py  # GNN implementation
│   │   └── model.py         # Prediction models
│   └── inference.py         # Inference script
├── 📁 example/              # Example data
│   ├── prot.pdb            # Example protein
│   ├── ligs.sdf            # Example ligands
│   └── run.sh              # Example script
├── 📁 save/                 # Pre-trained models
│   ├── reg.pth             # RMSD model weights
│   └── bce.pth             # Confidence model weights
├── setup.py                # Package configuration
├── env.yaml                # Conda environment
└── README.md               # This file
```

## 🔬 Example

### Complete Workflow

```bash
# Navigate to example directory
cd example

# Run prediction
bindingrmsd-inference \
    -r prot.pdb \
    -l ligs.sdf \
    -o binding_results.tsv \
    --batch_size 64 \
    --device cuda

# View results
head binding_results.tsv
```

### Expected Output

```
Name        pRMSD   Is_Above_2A ADG_Score
ligand_1    1.23    0.89        -8.5
ligand_2    3.45    0.12        -6.2
ligand_3    0.87    0.95        -9.1
...
```

## 🧪 Model Details

### Training Data
- Curated protein-ligand complexes with experimental binding poses
- Diverse chemical space coverage
- Quality-controlled RMSD annotations

### Model Architecture
- **Input**: Protein and ligand molecular graphs
- **Encoder**: Gated Graph Convolution with Local Structure-aware Positional Encoding
- **Output**: Regression (RMSD) + Binary Classification (Quality)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)  
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Citation

If you use BindingRMSD in your research, please cite:

```bibtex
@article{bindingrmsd2024,
  title={BindingRMSD: Accurate Prediction of Protein-Ligand Binding Pose RMSD using Graph Neural Networks},
  author={Jaemin Sim},
  year={2024}
}
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Jaemin Sim** - *Lead Developer* - [eightmm](https://github.com/eightmm)

## 🙏 Acknowledgments

- RDKit community for molecular informatics tools
- DGL team for graph neural network framework
- PyTorch team for deep learning infrastructure

---

<div align="center">

**[⭐ Star this repository](https://github.com/eightmm/BindingRMSD)** if you find it useful!

Made with ❤️ for the computational chemistry community

</div>

