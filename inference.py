import torch
<<<<<<< HEAD
import pandas as pd 

from tqdm import tqdm

from rdkit import Chem
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader

from model.model import PredictionRMSD
=======
import pandas as pd # type: ignore
from tqdm import tqdm
from rdkit import Chem # type: ignore
from dgl.dataloading import GraphDataLoader # type: ignore
from model.model import PredictionPKD
from data.data import PoseSelectionDataset
>>>>>>> change

def rmsd_inference_combined(protein_pdb, ligand_sdf, output, batch_size, model_path, device='cpu'):
    dataset = PoseSelectionDataset(
        protein_pdb=protein_pdb,
        ligand_sdf=ligand_sdf
    )

    loader = GraphDataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    rmsd_model = PredictionPKD(57, 256, 13, 25, 20, 4, 0).to(device)
    prob_model = PredictionPKD(57, 256, 13, 25, 20, 4, 0).to(device)

    reg_save_path = f'{model_path}/reg.pth'
    bce_save_path = f'{model_path}/bce.pth'

    rmsd_model.load_state_dict(torch.load(reg_save_path, weights_only=True)['model_state_dict'])
    prob_model.load_state_dict(torch.load(bce_save_path, weights_only=True)['model_state_dict'])

    rmsd_model.eval()
    prob_model.eval()

    results = {
        "Name": [],
        "RMSD": [],
        "Prob": [],
        "RMSD*Prob": [],
        "RMSD+Prob": [],
    }

    with torch.no_grad():
        progress_bar = tqdm(total=len(loader.dataset), unit='ligand')

        for data in loader:
            bgp, bgl, bgc, error, names = data
            bgp, bgl, bgc = bgp.to(device), bgl.to(device), bgc.to(device)

            rmsd = rmsd_model(bgp, bgl, bgc)
            prob = prob_model(bgp, bgl, bgc)

            rmsd = rmsd.view(-1)
            prob = prob.view(-1)

            prob = torch.sigmoid(prob)

            rmsd[error == 1] = torch.tensor(float('nan'))
            prob[error == 1] = torch.tensor(float('nan'))

<<<<<<< HEAD
    rmsd_model = PredictionRMSD(57, 256, 13, 25, 20, 4, 0.2).to(device)
    prob_model = PredictionRMSD(57, 256, 13, 25, 20, 4, 0.2).to(device)
=======
            results["Name"].extend(names)
            results["RMSD"].extend(rmsd.tolist())
            results["Prob"].extend(prob.tolist())
            results["RMSD*Prob"].extend((rmsd * prob).tolist())
            results["RMSD+Prob"].extend((rmsd + prob).tolist())
>>>>>>> change

            progress_bar.update(len(names))

        progress_bar.close()

    df = pd.DataFrame(results)
    df = df.round(4)
    df.to_csv(output, sep='\t', na_rep='NaN', index=False)

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
<<<<<<< HEAD
    parser.add_argument('-r','--protein_pdb', default='./example/1KLT_rec.pdb', help='receptor .pdb')
    parser.add_argument('-l','--ligand_sdf', default='./example/chk.sdf', help='ligand .sdf')
    parser.add_argument('-o','--output', default='./example/result.csv', help='result output file')
=======
    parser.add_argument('-r', '--protein_pdb', default='./1KLT_rec.pdb', help='receptor .pdb')
    parser.add_argument('-l', '--ligand_sdf', default='./chk.sdf', help='ligand .sdf')
    parser.add_argument('-o', '--output', default='./result.csv', help='result output file')
>>>>>>> change

    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--ncpu', default=4, type=int, help="cpu worker number")
    parser.add_argument('--device', type=str, default='cuda', help='choose device: cpu or cuda')
    parser.add_argument('--model_path', type=str, default='./save', help='model weight path')

    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.ncpu)
    os.environ["MKL_NUM_THREADS"] = str(args.ncpu)
    torch.set_num_threads(args.ncpu)

    if args.device == 'cpu':
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("gpu is not available, run on cpu")
            device = torch.device("cpu")

    rmsd_inference_combined(
        protein_pdb=args.protein_pdb,
        ligand_sdf=args.ligand_sdf,
        output=args.output,
        batch_size=args.batch_size,
        model_path=args.model_path,
        device=args.device
    )

