from copy import deepcopy
import torch
from torch_geometric.transforms import SVDFeatureReduction
from torch_geometric.datasets import Planetoid, WebKB, Amazon, WikipediaNetwork
from torch_geometric.data import Data
from torch_geometric.utils import degree, add_self_loops
import math

def x_padding(data, out_dim):
    
    assert data.x.size(-1) <= out_dim
    
    incremental_dimension = out_dim - data.x.size(-1)
    zero_features = torch.zeros((data.x.size(0), incremental_dimension), dtype=data.x.dtype, device=data.x.device)
    data.x = torch.cat([data.x, zero_features], dim=-1)

    return data


def x_svd(data, out_dim):
    
    assert data.x.size(-1) >= out_dim

    reduction = SVDFeatureReduction(out_dim)
    return reduction(data)



# including projection operation, SVD
def preprocess(data, node_feature_dim):

    if hasattr(data, 'train_mask'):
        del data.train_mask
    if hasattr(data, 'val_mask'):
        del data.val_mask
    if hasattr(data, 'test_mask'):
        del data.test_mask

    if node_feature_dim <= 0:
        edge_index_with_loops = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
        data.x = degree(edge_index_with_loops[1]).reshape((-1,1))
    
    else:
        # import pdb
        # pdb.set_trace()        
        if data.x.size(-1) > node_feature_dim:
            data = x_svd(data, node_feature_dim)
        elif data.x.size(-1) < node_feature_dim:
            data = x_padding(data, node_feature_dim)
        else:
            pass
    
    return data

# For prompting
def loss_contrastive_learning(x1, x2):
    # T = 0.1
    T = 0.5
    batch_size, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)
    
    sim_matrix = torch.einsum('ik,jk->ij', x1+1e-7, x2+1e-7) / torch.einsum('i,j->ij', x1_abs+1e-7, x2_abs+1e-7)
    
    if(True in sim_matrix.isnan()):
        print('Emerging nan value')
    
    sim_matrix = torch.exp(sim_matrix / T)
    
    if(True in sim_matrix.isnan()):
        print('Emerging nan value')    
    
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]

    if(True in pos_sim.isnan()):
        print('Emerging nan value')

    loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim) + 1e-4)
    loss = - torch.log(loss).mean()
    if math.isnan(loss.item()):
        print("The value is NaN.")

    return loss

# used in pre_train.py
def gen_ran_output(data, simgrace, reconstruct):
    vice_model = deepcopy(simgrace)

    for (vice_name, vice_model_param), (name, param) in zip(vice_model.named_parameters(), simgrace.named_parameters()):
        if vice_name.split('.')[0] == 'projection_head':
            vice_model_param.data = param.data
        else:
            vice_model_param.data = param.data + 0.1 * torch.normal(0, torch.ones_like(
                param.data) * param.data.std())
    if(reconstruct==0.0):
    
        zj = vice_model.forward_cl(data)

        return zj
    
    else:
    
        zj, hj = vice_model.forward_cl(data)

        return zj, hj