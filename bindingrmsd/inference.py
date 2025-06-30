import torch
import pandas as pd 

from tqdm import tqdm

from dgl.dataloading import GraphDataLoader

from bindingrmsd.data.data import PoseSelectionDataset
from bindingrmsd.model.model import PredictionRMSD


def inference(protein_pdb, ligand_file, output, batch_size, model_path, device='cpu'):
    dataset = PoseSelectionDataset(
        protein_pdb=protein_pdb,
        ligand_file=ligand_file
    )

    loader = GraphDataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    rmsd_model = PredictionRMSD(57, 256, 13, 25, 20, 4, 0).to(device)
    prob_model = PredictionRMSD(57, 256, 13, 25, 20, 4, 0).to(device)

    reg_save_path = f'{model_path}/reg.pth'
    bce_save_path = f'{model_path}/bce.pth'

    rmsd_model.load_state_dict(torch.load(reg_save_path, weights_only=True)['model_state_dict'])
    prob_model.load_state_dict(torch.load(bce_save_path, weights_only=True)['model_state_dict'])

    rmsd_model.eval()
    prob_model.eval()

    results = {
        "Name": [],
        "pRMSD": [],
        "Is_Above_2A": [],
        "ADG_Score": [],
    }

    with torch.no_grad():
        progress_bar = tqdm(total=len(loader.dataset), unit='ligand')

        for data in loader:
            bgp, bgl, bgc, error, names, adg_score = data
            bgp, bgl, bgc = bgp.to(device), bgl.to(device), bgc.to(device)

            rmsd = rmsd_model(bgp, bgl, bgc)
            prob = prob_model(bgp, bgl, bgc)

            rmsd = rmsd.view(-1)
            prob = prob.view(-1)

            prob = torch.sigmoid(prob)

            rmsd[error == 1] = torch.tensor(float('nan'))
            prob[error == 1] = torch.tensor(float('nan'))

            results["Name"].extend(names)
            results["pRMSD"].extend(rmsd.tolist())
            results["Is_Above_2A"].extend(prob.tolist())
            results["ADG_Score"].extend(adg_score.tolist())
            progress_bar.update(len(names))

        progress_bar.close()

    df = pd.DataFrame(results)
    df = df.round(4)
    df.to_csv(output, sep='\t', na_rep='NaN', index=False)

def main():
    """Main entry point for the command line interface."""
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description='BindingRMSD: Predict protein-ligand binding pose RMSD using Graph Neural Networks'
    )
    parser.add_argument(
        '-r', '--protein_pdb', 
        required=True,
        help='Receptor protein PDB file'
    )
    parser.add_argument(
        '-l', '--ligand_file', 
        required=True,
        help='Ligand file (.sdf, .mol2, .dlg, .pdbqt, or .txt list)'
    )
    parser.add_argument(
        '-o', '--output', 
        default='./result.tsv',
        help='Output results file (default: result.tsv)'
    )
    parser.add_argument(
        '--batch_size', 
        default=128, 
        type=int,
        help='Batch size for inference (default: 128)'
    )
    parser.add_argument(
        '--ncpu', 
        default=4, 
        type=int,
        help="Number of CPU workers (default: 4)"
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda',
        choices=['cpu', 'cuda'],
        help='Compute device: cpu or cuda (default: cuda)'
    )
    parser.add_argument(
        '--model_path', 
        type=str, 
        default='./save',
        help='Directory containing model weights (default: ./save)'
    )

    args = parser.parse_args()

    # Set threading configuration
    os.environ["OMP_NUM_THREADS"] = str(args.ncpu)
    os.environ["MKL_NUM_THREADS"] = str(args.ncpu)
    torch.set_num_threads(args.ncpu)

    # Device selection with better error handling
    if args.device == 'cpu':
        device = torch.device("cpu")
        print("🖥️  Using CPU for inference")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"🚀 Using GPU for inference: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  GPU is not available, switching to CPU")
            device = torch.device("cpu")

    # Run inference
    print(f"🧬 Starting BindingRMSD inference...")
    print(f"📁 Protein: {args.protein_pdb}")
    print(f"📁 Ligands: {args.ligand_file}")
    print(f"📁 Output: {args.output}")
    print(f"⚙️  Batch size: {args.batch_size}")
    print()

    inference(
        protein_pdb=args.protein_pdb,
        ligand_file=args.ligand_file,
        output=args.output,
        batch_size=args.batch_size,
        model_path=args.model_path,
        device=device
    )

    print(f"✅ Inference completed! Results saved to {args.output}")


if __name__ == "__main__":
    main()

