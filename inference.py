import os
import torch
import pandas as pd 

from tqdm import tqdm

from rdkit import Chem
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader

from model.model import PredictionRMSD

from data.ligand_atom_feature import mol_to_graph
from data.protein_atom_feature import get_all_graph, prot_to_graph, pl_to_c_graph

os.environ['PATH'] = '/usr/local/cuda-12.1/bin:' + os.environ['PATH']
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-12.1/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')


class PoseSelectionDataset(DGLDataset):
    def __init__(self, protein_pdb, ligand_sdf):
        super(PoseSelectionDataset, self).__init__(name='Protein Ligand Binding conformation RMSD prediction')

        self.ligand_mols = Chem.SDMolSupplier( ligand_sdf )
        self.gp = prot_to_graph( protein_pdb )

    def __getitem__(self, idx):
        try:
            gl = mol_to_graph( self.ligand_mols[idx] )
            gp, gl, gc = get_all_graph( self.gp, gl )
            error = 0
            
        except:
            gl = self.lig_dummy_graph(num_nodes=3)
            gp, gl, gc = get_all_graph( self.gp, gl)
            error = 1
        return gp, gl, gc, error, idx

    def __len__(self):
        return len(self.ligand_mols)

    def lig_dummy_graph(self, num_nodes):
        src = torch.randint(0, num_nodes, (10,))
        dst = torch.randint(0, num_nodes, (10,))
        gl = dgl.graph( (src, dst), num_nodes=num_nodes)
        gl.ndata['feat'] = torch.zeros((num_nodes, 57)).float()  # Example: adding dummy node features
        gl.ndata['pos_enc'] = torch.zeros((num_nodes, 20)).float()  # Example: adding dummy node features
        gl.ndata['coord'] = torch.randn((num_nodes, 3)).float()  # Example: adding dummy node features
        gl.edata['feat'] = torch.zeros((10, 13)).float()
        return gl



def inference(model1, model2, loader, output, device='cpu'):
    model1.eval()
    model2.eval()

    results = {
        "Name": [],
        "RMSD": [],
        "Prob": [],
        "RMSD*Prob":[],
        "RMSD+Prob":[],
    }
    
    with torch.no_grad():
        progress_bar = tqdm( total=len(loader.dataset), unit='ligand' )

        for data in loader:
            
            bgp, bgl, bgc, error, idx = data
            bgp, bgl, bgc = bgp.to(device), bgl.to(device), bgc.to(device)

            rmsd = model1(bgp, bgl, bgc)
            prob = model2(bgp, bgl, bgc)

            rmsd = rmsd.view(-1)
            prob = prob.view(-1)

            prob = torch.sigmoid(prob)

            rmsd[error==1] = torch.tensor(float('nan'))
            prob[error==1] = torch.tensor(float('nan'))

            results["Name"].extend( [ str(int(i)) for i in idx ] )
            results['RMSD'].extend( rmsd.tolist() )
            results['Prob'].extend( prob.tolist() )
            results['RMSD*Prob'].extend( (rmsd * prob).tolist() )
            results['RMSD+Prob'].extend( (rmsd + prob).tolist() )
            
            progress_bar.update(len(idx))
        
        progress_bar.close()

    df = pd.DataFrame( results )
    df = df.round(4)
    df.to_csv(output, sep='\t', na_rep='NaN', index=False)


def rmsd_prediction( protein_pdb, ligand_sdf, output, batch_size, model_path, device ):
    dataset = PoseSelectionDataset( 
        protein_pdb=protein_pdb,
        ligand_sdf=ligand_sdf
    )

    loader = GraphDataLoader( dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    rmsd_model = PredictionRMSD(57, 256, 13, 25, 20, 4, 0.2).to(device)
    prob_model = PredictionRMSD(57, 256, 13, 25, 20, 4, 0.2).to(device)

    reg_save_path = f'{model_path}/reg.pth'
    bce_save_path = f'{model_path}/bce.pth'

    rmsd_model.load_state_dict(torch.load(reg_save_path)['model_state_dict'])
    prob_model.load_state_dict(torch.load(bce_save_path)['model_state_dict'])

    inference( rmsd_model, prob_model, loader, output, device )

if __name__ == "__main__" :
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--protein_pdb', default='./example/1KLT_rec.pdb', help='receptor .pdb')
    parser.add_argument('-l','--ligand_sdf', default='./example/chk.sdf', help='ligand .sdf')
    parser.add_argument('-o','--output', default='./example/result.csv', help='result output file')

    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--ncpu', default=4, type=int, help="cpu worker number")
    parser.add_argument('--device', type=str, default='cuda', help='choose device: cpu or cuda')
    parser.add_argument('--model_path', type=str, default='./save', help='model weight path')

    args = parser.parse_args()

    if args.device == 'cpu':
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("gpu is not available, run on cpu")
            device = torch.device("cpu")

    rmsd_prediction( 
        protein_pdb=args.protein_pdb, 
        ligand_sdf=args.ligand_sdf,
        output=args.output,
        batch_size=args.batch_size,
        model_path=args.model_path,
        device=args.device
    )
