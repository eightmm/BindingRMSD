import torch, dgl
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from dgl.nn.pytorch.glob import AvgPooling, SumPooling, MaxPooling
from dgl.nn.pytorch.conv import EGNNConv

from .GatedGCNLSPE import GatedGCNLSPELayer
from .GraphGPS import GraphGPS

class PredictionPKD(nn.Module):
    def __init__(self, in_size, emb_size, intra_edge_size, inter_edge_size, pose_size, num_layers, dropout_ratio=0.15):
        super(PredictionPKD, self).__init__()
        self.res_token_encoder  = nn.Embedding( 22, int(emb_size / 2) )
        self.atom_token_encoder = nn.Embedding( 176, int(emb_size / 2) )
        self.protein_edge_encoder  = nn.Linear( 15, emb_size )

        self.ligand_node_encoder = nn.Linear( in_size, emb_size )
        self.ligand_edge_encoder = nn.Linear( intra_edge_size,  emb_size )

        self.protein_pose_encoder = nn.Linear( pose_size, emb_size )
        self.ligand_pose_encoder  = nn.Linear( pose_size, emb_size )

        self.complex_edge_encoder = nn.Linear( 15, emb_size )

        self.protein_norm = nn.LayerNorm( emb_size )
        self.ligand_norm  = nn.LayerNorm( emb_size )

        blocks = [
            nn.ModuleList(
                [
                    # GraphGPS(
                    #     emb_size,
                    #     8
                    # )

                    # GatedGCNLSPELayer(
                    #     input_dim=emb_size,
                    #     output_dim=emb_size,
                    #     dropout=0.2,
                    #     batch_norm=True
                    # )

                    EGNNConv(
                        in_size=emb_size,
                        hidden_size=emb_size,
                        out_size=emb_size,
                        edge_feat_size=15,
                    )

                    for _ in range(num_layers)
                ]
            )
            for _ in range(3)
        ]

        self.protein_block = blocks[0]
        self.ligand_block  = blocks[1]
        self.complex_block = blocks[2]

        self.mlp_binding_affinity = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.BatchNorm1d(emb_size),
            nn.ELU(),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(emb_size, 1),
        )

        self.sum_pooling = SumPooling()
        self.max_pooling = MaxPooling()

    def forward(self, gp, gl, gc):
        hpr = self.res_token_encoder( gp.ndata['token_res'] )
        hpa = self.atom_token_encoder( gp.ndata['token_atom'])

        hp = torch.cat( [hpr, hpa], 1 )

        ep = self.protein_edge_encoder( gp.edata['dist'] )
        pp = self.protein_pose_encoder( gp.ndata['pos_enc'] )

        hl = self.ligand_node_encoder( gl.ndata['feat'] )
        el = self.ligand_edge_encoder( gl.edata['feat'] )
        pl = self.ligand_pose_encoder( gl.ndata['pos_enc'])

        ec = self.complex_edge_encoder( gc.edata['dist'] )

        hp = self.protein_norm( hp )
        hl = self.ligand_norm( hl )

        xp = gp.ndata['coord']
        xl = gl.ndata['coord']
        print(xp.shape)
        print(xl.shape)
        xc = torch.cat( (xp, xl) )
        

        print(xc.shape)
        hp_raw = hp
        hl_raw = hl

        gp_batch_sizes = gp.batch_num_nodes()
        gl_batch_sizes = gl.batch_num_nodes()

        gp_start_indices = [0] + torch.cumsum(gp_batch_sizes[:-1], dim=0).tolist()
        gl_start_indices = [0] + torch.cumsum(gl_batch_sizes[:-1], dim=0).tolist()

        for (protein_layer, ligand_layer, complex_layer) in zip(self.protein_block, self.ligand_block, self.complex_block):
            # hp, pp, ep = protein_layer( gp, hp, pp, ep ) #  g, h, p, e,
            # hl, pl, el = ligand_layer( gl, hl, pl, el )
            hp, _ = protein_layer( gp, hp, xp, ep )
            hl, _ = ligand_layer( gl, hl, xl, el )

            hc = []
            pc = []
            for gp_start, gp_size, gl_start, gl_size in zip(gp_start_indices, gp_batch_sizes, gl_start_indices, gl_batch_sizes):
                gp_slice = hp[gp_start:gp_start + gp_size]
                gl_slice = hl[gl_start:gl_start + gl_size]
                pp_slice = pp[gp_start:gp_start + gp_size]
                pl_slice = pl[gl_start:gl_start + gl_size]
                hc.append( torch.cat( [gp_slice, gl_slice] ) )
                pc.append( torch.cat( [pp_slice, pl_slice] ) )

            hc = torch.cat( hc )
            pc = torch.cat( pc )

            # hc, pc, ec = complex_layer( gc, hc, pc, ec )
            hc, _ = complex_layer( gc, hc, xc, ec )


            hp_separated = []
            hl_separated = []
            start = 0
            for gp_size, gl_size in zip(gp_batch_sizes, gl_batch_sizes):
                hp_separated.append(hc[start: start + gp_size])
                start += gp_size
                hl_separated.append(hc[start: start + gl_size])
                start += gl_size

            hp = torch.cat(hp_separated)
            hl = torch.cat(hl_separated)

        # h = self.sum_pooling(gl, hl)
        h = self.sum_pooling(gc, hc)

        binding_affinity = self.mlp_binding_affinity( h )

        return binding_affinity
