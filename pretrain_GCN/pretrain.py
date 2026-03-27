import numpy as np
import torch
from copy import deepcopy
import random
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import argparse

from graph_augment import graph_views
from contrastive import get_clustered_data
from get_model import get_model

def run(args, trial=None):

    if(args.saliency_model == 'mlp'):

        data = get_clustered_data(args.dataset, args.cache_dir, args.cross_link, args.node_feature_dim, args.cl_init_method, args.cross_link_ablation, args.dynamic_edge, args.dynamic_prune, args.split_method)

        model = get_model(
            backbone_kwargs = {
                'name': args.backbone_model,
                'num_features': data[0].x.size(-1),
                'hid_dim': args.hid_dim,
                'num_conv_layers': args.num_conv_layers,
                'dropout': args.dropout,
                'reconstruct': args.reconstruct,
            },
            saliency_kwargs = {
                'name': args.saliency_model,
                'feature_dim': data[0].x.size(-1),
            } if args.saliency_model != 'none' else None,
        )
    else:

        with torch.no_grad():
            data, gco_model, raw_data = get_clustered_data(args.dataset, args.cache_dir, args.cross_link, args.node_feature_dim, args.cl_init_method, args.cross_link_ablation, args.dynamic_edge, args.dynamic_prune, args.split_method)

        # init model
        # from get_model import get_model
        model = get_model(
            backbone_kwargs = {
                'name': args.backbone_model,
                'num_features': data[0].x.size(-1),
                'hid_dim': args.hid_dim,
                'num_conv_layers': args.num_conv_layers,
                'dropout': args.dropout,
                'reconstruct': args.reconstruct,
            },
            saliency_kwargs = {
                'name': args.saliency_model,
                'feature_dim': data[0].x.size(-1),
            } if args.saliency_model != 'none' else None,
        )
    
    # train
    if args.method == 'graphcl':
        model, loss = graph_cl_pretrain(data, model, gco_model, raw_data, args.learning_rate,
                    args.weight_decay, args.epoch, args.cross_link, args.cl_init_method,
                    args.reconstruct, args.dynamic_edge, args.split_method, args.batch_size)
    elif args.method == 'simgrace':
        model, loss = simgrace_pretrain(data, model, gco_model, raw_data,
                    args.weight_decay, args.epoch, args.cross_link, args.cl_init_method,
                    args.reconstruct, args.dynamic_edge, args.split_method, args.batch_size)
    else:
        raise NotImplementedError(f'Unknown method: {args.method}')

    # save
    import os

    return loss

def graph_cl_pretrain(
    data,
    model,
    gco_model,
    raw_data,
    learning_rate,
    weight_decay,
    epoch,
    cross_link,
    cl_init_method,
    reconstruct,
    dynamic_edge,
    split_method,
    batch_size,
    ):

    def get_loaders(data, batch_size):

        augs, aug_ratio = random.choices(['dropN', 'permE', 'maskN'], k=2), random.randint(1, 3) * 1.0 / 10

        view_list_1 = []
        view_list_2 = []
        for g in data:
            view_g = graph_views(data=g, aug=augs[0], aug_ratio=aug_ratio)
            view_list_1.append(Data(x=view_g.x, edge_index=view_g.edge_index))
            view_g = graph_views(data=g, aug=augs[1], aug_ratio=aug_ratio)
            view_list_2.append(Data(x=view_g.x, edge_index=view_g.edge_index))

        loader1 = DataLoader(view_list_1, batch_size=batch_size, shuffle=False,
                                num_workers=4)  
        loader2 = DataLoader(view_list_2, batch_size=batch_size, shuffle=False,
                                num_workers=4)  

        return loader1, loader2

    class ContrastiveLoss(torch.nn.Module):
        def __init__(self, hidden_dim, temperature=0.5):
            super(ContrastiveLoss, self).__init__()
            self.head = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )
            self.temperature = temperature

        def forward(self, zi, zj):
            batch_size = zi.size(0)
            x1_abs = zi.norm(dim=1)
            x2_abs = zj.norm(dim=1)
            sim_matrix = torch.einsum('ik,jk->ij', zi, zj) / torch.einsum('i,j->ij', x1_abs, x2_abs)
            sim_matrix = torch.exp(sim_matrix / self.temperature)
            pos_sim = sim_matrix[range(batch_size), range(batch_size)]
            loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss = - torch.log(loss).mean()
            return loss

    class ReconstructionLoss(torch.nn.Module):
        def __init__(self, hidden_dim, feature_num):
            super(ReconstructionLoss, self).__init__()
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden_dim, feature_num),
            )

            self.loss_fn = torch.nn.MSELoss()

        def forward(self, input_features, hidden_features):
            reconstruction_features = self.decoder(hidden_features)
            return self.loss_fn(input_features, reconstruction_features)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   
    
    loss_fn = ContrastiveLoss(model.backbone.hidden_dim).to(device)
    loss_fn.train(), model.to(device).train()
    best_loss = 100000.
    best_model = None
    if(gco_model==None):
        if(reconstruct==0.0):
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(model.parameters()) + list(loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )
        else:
            rec_loss_fn = ReconstructionLoss(model.backbone.hidden_dim, data[0].num_node_features).to(device)
            rec_loss_fn.train()
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(model.parameters()) + list(loss_fn.parameters()) +list(rec_loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )                
    else:
        if(reconstruct==0.0):
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(gco_model.parameters()) + list(model.parameters()) + list(loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )
        else:
            rec_loss_fn = ReconstructionLoss(model.backbone.hidden_dim, data[0].num_node_features).to(device)
            rec_loss_fn.train()
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(gco_model.parameters()) + list(model.parameters()) + list(loss_fn.parameters()) +list(rec_loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )            

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)

    from torchmetrics import MeanMetric
    from tqdm import tqdm
    from contrastive import update_graph_list_param
    loss_metric = MeanMetric()

    for e in range(epoch):
        
        loss_metric.reset()

        if(cross_link > 0 and cl_init_method == 'learnable'):
            if(split_method=='RandomWalk'):
                last_updated_data = deepcopy(data)

            loaders = get_loaders(data, batch_size)
        elif(e==0):
            loaders = get_loaders(data, batch_size)

        pbar = tqdm(zip(*loaders), total=len(loaders[0]), ncols=100, desc=f'Epoch {e}, Loss: inf')
                
        for batch1, batch2 in pbar:

            if(gco_model!=None):
                batch1 = gco_model(batch1)
                batch2 = gco_model(batch2)    

            optimizer.zero_grad()

            if(reconstruct==0.0):
                zi, zj = model(batch1.to(device)), model(batch2.to(device))
                loss = loss_fn(zi, zj)
            else:               
                zi, hi = model(batch1.to(device))
                zj, hj = model(batch2.to(device))
                loss = loss_fn(zi, zj) + reconstruct*(rec_loss_fn(batch1.x, hi) + rec_loss_fn(batch2.x, hj))
                
            loss.backward()
            optimizer.step()
            
            loss_metric.update(loss.item(), batch1.size(0))
            pbar.set_description(f'Epoch {e}, Loss {loss_metric.compute():.4f}', refresh=True)

        if(gco_model!=None):
            data  = update_graph_list_param(last_updated_data, gco_model)
            gco_model.update_last_params()

        # lr_scheduler.step()
        
        if(loss_metric.compute()<best_loss):
            best_loss = loss_metric.compute()
            best_model = deepcopy(model)
            
        pbar.close()
        
    return best_model, best_loss

def simgrace_pretrain(
    data,
    model,
    gco_model,
    raw_data,
    learning_rate,
    weight_decay,
    epoch,
    cross_link,
    cl_init_method,
    reconstruct,
    dynamic_edge,
    split_method,
    batch_size,
    ):

    from torch_geometric.loader import DataLoader
    from utils import gen_ran_output

    class SimgraceLoss(torch.nn.Module):
        def __init__(self, gnn, hidden_dim, reconstruct, temperature=0.5):
            super(SimgraceLoss, self).__init__()
            self.gnn = gnn
            self.projection_head = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )
            self.temperature = temperature
            self.reconstruct = reconstruct

        def forward_cl(self, data):
        
            if(self.reconstruct==0.0):
                zi = self.gnn(data)
                zi = self.projection_head(zi)

                return zi
            else:
                zi, hi = self.gnn(data)
                zi = self.projection_head(zi)
            
                return zi, hi
        
        def loss_cl(self, zi, zj):
            batch_size = zi.size(0)
            x1_abs = zi.norm(dim=1)
            x2_abs = zj.norm(dim=1)
            sim_matrix = torch.einsum('ik,jk->ij', zi, zj) / torch.einsum('i,j->ij', x1_abs, x2_abs)
            sim_matrix = torch.exp(sim_matrix / self.temperature)
            pos_sim = sim_matrix[range(batch_size), range(batch_size)]
            loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss = - torch.log(loss).mean()
            return loss

    class ReconstructionLoss(torch.nn.Module):
        def __init__(self, hidden_dim, feature_num):
            super(ReconstructionLoss, self).__init__()
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden_dim, feature_num),
            )

            self.loss_fn = torch.nn.MSELoss()

        def forward(self, input_features, hidden_features):
            reconstruction_features = self.decoder(hidden_features)
            return self.loss_fn(input_features, reconstruction_features)


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   


    loss_fn = SimgraceLoss(model.backbone, model.backbone.hidden_dim, reconstruct).to(device)
    loss_fn.train(), model.to(device).train()
    best_loss = np.inf
    best_model = None
    if(gco_model==None):
        if(reconstruct==0.0):
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(model.parameters()) + list(loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )
        else:
            rec_loss_fn = ReconstructionLoss(model.backbone.hidden_dim, data[0].num_node_features).to(device)
            rec_loss_fn.train()
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(model.parameters()) + list(loss_fn.parameters()) +list(rec_loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )                
    else:
        if(reconstruct==0.0):
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(gco_model.parameters()) + list(model.parameters()) + list(loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )
        else:
            rec_loss_fn = ReconstructionLoss(model.backbone.hidden_dim, data[0].num_node_features).to(device)
            rec_loss_fn.train()
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(gco_model.parameters()) + list(model.parameters()) + list(loss_fn.parameters()) +list(rec_loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )            

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)

    from torchmetrics import MeanMetric
    from tqdm import tqdm
    from contrastive import update_graph_list_param
    loss_metric = MeanMetric()

    for e in range(epoch):
        
        loss_metric.reset()

        if(cross_link > 0 and cl_init_method == 'learnable'):
            if(split_method=='RandomWalk'):
                last_updated_data = deepcopy(data)

            loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=1) 
        elif(e==0):
            loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=1) 

        pbar = tqdm(loader, total=len(loader), ncols=100, desc=f'Epoch {e}, Loss: inf')

        for batch1 in pbar:
            if(gco_model!=None):
                batch1 = gco_model(batch1)

            optimizer.zero_grad()

            if(reconstruct==0.0):
                batch1 = batch1.to(device)
                zi = loss_fn.forward_cl(batch1)
                zj = gen_ran_output(batch1, loss_fn)
                zj = zj.detach().data.to(device)
                loss = loss_fn.loss_cl(zi, zj)              
            else:

                batch1 = batch1.to(device)
                zi, hi = loss_fn.forward_cl(batch1)
                zj, hj = gen_ran_output(batch1, loss_fn)
                zj = zj.detach().data.to(device)
                loss = loss_fn.loss_cl(zi, zj) + reconstruct*(rec_loss_fn(batch1.x, hi) + rec_loss_fn(batch1.x, hj))
                
            loss.backward()
            optimizer.step()
            
            loss_metric.update(loss.item(), batch1.size(0))
            pbar.set_description(f'Epoch {e}, Loss {loss_metric.compute():.4f}', refresh=True)

        if(gco_model!=None):
            data  = update_graph_list_param(last_updated_data, gco_model)
            gco_model.update_last_params()

        # lr_scheduler.step()
        
        if(loss_metric.compute()<best_loss):
            best_loss = loss_metric.compute()
            best_model = deepcopy(model)
            
        pbar.close()
        
    return best_model

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Pretrain')
    parser.add_argument("--save_dir", type=str, default="../storage/chengdu_train_beijing_test",
                        help="directory to save results")
    parser.add_argument("--dataset", type=str, nargs="+", default=["chengdu", "proto"], help="dataset names")
    parser.add_argument("--backbone_model", type=str, default="gcn", help="backbone model type")
    parser.add_argument("--saliency_model", type=str, default="none", help="saliency model type")
    parser.add_argument("--method", type=str, default="graphcl", help="training method")
    parser.add_argument("--noise_switch", action="store_true", help="whether to enable noise switch")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.2, help="weight decay")
    parser.add_argument("--epoch", type=int, default=50, help="number of training epochs")
    parser.add_argument("--cross_link", type=int, default=1, help="number of cross links")
    parser.add_argument("--cl_init_method", type=str, default="learnable", help="cross link initialization method")
    parser.add_argument("--cross_link_ablation", action="store_true", help="whether to perform cross-link ablation")
    parser.add_argument("--reconstruct", type=float, default=0.2, help="reconstruction loss weight")
    parser.add_argument("--dynamic_edge", type=str, default="none", help="dynamic edge type")
    parser.add_argument("--dynamic_prune", type=float, default=0.2, help="dynamic pruning ratio")
    parser.add_argument("--split_method", type=str, default="RandomWalk", help="graph split method")
    parser.add_argument("--batch_size", type=int, default=32, help="training batch size")
    parser.add_argument("--cache_dir", type=str, default="../storage/.cache", help="cache directory")
    parser.add_argument("--node_feature_dim", type=int, default=512, help="dimension of node features")
    parser.add_argument("--hid_dim", type=int, default=32, help="hidden dimension")
    parser.add_argument("--num_conv_layers", type=int, default=2, help="number of GCN layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")

    args = parser.parse_args()


    run(args)
