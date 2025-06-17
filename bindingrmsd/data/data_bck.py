import os, torch, dgl

from rdkit import Chem # type: ignore
from dgl.data import DGLDataset # type: ignore

from .ligand_atom_feature import mol_to_graph
from .protein_atom_feature import get_all_graph, prot_to_graph, pl_to_c_graph

def process_dlg(file_path, only_cluster=False):
    from meeko import PDBQTMolecule
    from meeko import RDKitMolCreate

    name = os.path.basename(file_path).split('.')[0]
    if '.dlg' in file_path:
        pdbqt_mol = PDBQTMolecule.from_file(file_path, name=name, is_dlg=True, skip_typing=True)
        
    elif '.pdbqt' in file_path:
        pdbqt_mol = PDBQTMolecule.from_file(file_path, name=name, skip_typing=True)

    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    rdkit_mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol, only_cluster_leads=only_cluster, keep_flexres=False)
    sdf_string, _ = RDKitMolCreate.write_sd_string( pdbqt_mol, only_cluster_leads=only_cluster )

    adg_score = []
    for line in sdf_string.split('\n'):
        if '{' in line:
            words = line.split(',')
            free_energy = words[1].split(':')[1].strip()
            adg_score.append(float(free_energy))
    
    mols = []
    err_tags = []
    names = []
    for i, conf in enumerate(rdkit_mols[0].GetConformers()):
        mol = Chem.Mol(rdkit_mols[0])
        if mol is None:
            mols.append(None)
            err_tags.append(1)
            names.append(f"{name}_{i}")
            continue
        else:
            mol.RemoveAllConformers()
            mol.AddConformer(conf, assignId=True)
            mols.append(mol)
            err_tags.append(0)
            names.append(f"{name}_{i}")
    # print(mols, err_tags, names, adg_score)
    return mols, err_tags, names, adg_score

def process_ligand_file(file_path):
    extension = os.path.splitext(file_path)[-1].lower()

    if extension == '.sdf':
        supplier = enumerate(Chem.SDMolSupplier(file_path, sanitize=False))
    elif extension == '.mol2':
        with open(file_path, 'r') as f:
            mol2_data = f.read()
        mol2_blocks = mol2_data.split('@<TRIPOS>MOLECULE')
        supplier = enumerate(
            Chem.MolFromMol2Block('@<TRIPOS>MOLECULE' + block, sanitize=False) 
            for block in mol2_blocks[1:]
        )
    else:
        raise ValueError(f"Unsupported file type: {extension}")

    ligands = []
    err_tag = []
    ligand_names = []
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    for idx, mol in supplier:
        if mol is not None:
            ligands.append(mol)
            err_tag.append(0)
            ligand_name = mol.GetProp('_Name') if mol.HasProp('_Name') else ''
            if ligand_name == '':
                ligand_name = f"{base_name}_{idx}"
            ligand_names.append(ligand_name)
        else:
            ligands.append(None)
            err_tag.append(1)
            ligand_names.append(f"{base_name}_{idx}")

    return ligands, err_tag, ligand_names


def load_ligands(file_path):
    lig_mols = []
    err_tags = []
    lig_names = []

    def process_single_file(line):
        assert os.path.isfile(line), f"File not found: {line}"
        return process_ligand_file(line)

    file_extension = os.path.splitext(file_path)[-1].lower()

    if file_extension == '.txt':
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:
            file_ligands, file_err_tag, file_ligand_names = process_single_file(line)
            lig_mols.extend(file_ligands)
            err_tags.extend(file_err_tag)
            lig_names.extend(file_ligand_names)

    elif file_extension in ['.sdf', '.mol2']:
        lig_mols, err_tags, lig_names = process_single_file(file_path)

    else:
        raise ValueError("Unsupported file type. Use '.txt', '.sdf', or '.mol2'.")

    return lig_mols, err_tags, lig_names


class PoseSelectionDataset(DGLDataset):
    def __init__(self, protein_pdb, ligand_file):
        super(PoseSelectionDataset, self).__init__(name='Protein Ligand Binding conformation RMSD prediction')

        # self.ligand_mols = Chem.SDMolSupplier( ligand_sdf )
        self.lig_mols, self.err_tags, self.lig_names = load_ligands(ligand_file)
        self.gp = prot_to_graph( protein_pdb )

    def __getitem__(self, idx):
        try:
            mol = self.lig_mols[idx]
            gl = mol_to_graph(mol)
            gp, gl, gc = get_all_graph(self.gp, gl)
            error = self.err_tags[idx]
            name = self.lig_names[idx]

        except:
            gl = self.lig_dummy_graph(num_nodes=3)
            gp, gl, gc = get_all_graph(self.gp, gl)
            error = self.err_tags[idx]
            name = self.lig_names[idx]

        return gp, gl, gc, error, name

    def __len__(self):
        return len(self.lig_mols)

    def lig_dummy_graph(self, num_nodes):
        src = torch.randint(0, num_nodes, (10,))
        dst = torch.randint(0, num_nodes, (10,))
        gl = dgl.graph( (src, dst), num_nodes=num_nodes)
        gl.ndata['feat'] = torch.zeros((num_nodes, 57)).float()  # Example: adding dummy node features
        gl.ndata['pos_enc'] = torch.zeros((num_nodes, 20)).float()  # Example: adding dummy node features
        gl.ndata['coord'] = torch.randn((num_nodes, 3)).float()  # Example: adding dummy node features
        gl.edata['feat'] = torch.zeros((10, 13)).float()
        return gl

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dlg', type=str, required=True)
    parser.add_argument('--only_cluster', action='store_true')
    args = parser.parse_args()
    process_dlg(args.dlg, args.only_cluster)