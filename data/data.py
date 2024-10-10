import os
import torch

from rdkit import Chem # type: ignore
from dgl.data import DGLDataset # type: ignore

from .ligand_atom_feature import mol_to_graph
from .protein_atom_feature import get_all_graph, prot_to_graph, pl_to_c_graph


class PoseSelectionDataset(DGLDataset):
    def __init__(self, protein_pdb, ligand_sdf):
        super(PoseSelectionDataset, self).__init__(name='Protein Ligand Binding conformation RMSD prediction')

        self.ligand_mols = Chem.SDMolSupplier( ligand_sdf )
        self.gp = prot_to_graph( protein_pdb )

    def __getitem__(self, idx):
        try:
            mol = self.ligand_mols[idx]
            gl = mol_to_graph(mol)
            gp, gl, gc = get_all_graph(self.gp, gl)
            error = 0

            name = mol.GetProp('_Name') 
            if name == '':
               name = str(idx)

        except:
            gl = self.lig_dummy_graph(num_nodes=3)
            gp, gl, gc = get_all_graph(self.gp, gl)
            error = 1
            name = str(idx)  # 에러가 발생하면 idx 사용

        return gp, gl, gc, error, name

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

