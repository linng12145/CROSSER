import torch
from torch import nn


from torch_geometric.data import Data


from pretrain_GCN.gcn import get_model as get_pretrained_backbone_gcn


def _adj_to_edge_index(adj):

    if adj.is_sparse:
        # torch.sparse_coo_tensor
        adj = adj.coalesce()
        return adj.indices()
    else:
        # dense tensor
        idx = (adj > 0).nonzero(as_tuple=False).t()
        return idx


class PretrainedGCNAdapter(nn.Module):
    def __init__(self,
                 ckpt_path: str,
                 num_features: int,
                 hid_dim: int,
                 num_conv_layers: int = 3,
                 dropout: float = 0.0,
                 reconstruct: float = 1.0, 
                 map_location: str = "cpu"):
        super().__init__()

        self.backbone = get_pretrained_backbone_gcn(
            num_features=num_features,
            hid_dim=hid_dim,
            num_conv_layers=num_conv_layers,
            dropout=dropout,
            reconstruct=reconstruct,
        )


        state = torch.load(ckpt_path, map_location=map_location)
        sd = state.get("model", state) 
        backbone_sd = {k.replace("backbone.", ""): v
                       for k, v in sd.items() if k.startswith("backbone.")}

        self.backbone.load_state_dict(backbone_sd, strict=False)
        self.backbone.eval()  
        for param in self.parameters():
            param.requires_grad = True

    @torch.no_grad()
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:

        device = x.device
        edge_index = _adj_to_edge_index(adj).to(device)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device) 

        data = Data(x=x, edge_index=edge_index, batch=batch)
        out = self.backbone(data)  

        if isinstance(out, tuple):
            _, h = out
            return h
        else:
            raise RuntimeError(
                "reconstruct!=0"
            )
