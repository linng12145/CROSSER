import argparse
import warnings
import time
import os
import torch
import torch.nn as nn
from model import Transformer_tagging, CL_Loss
from dataloader import TrajectoryTaggingDataset, TestingTaggingDataset, dataloader_collate, dataloader_collate_test

from utils import *
from constants import *
warnings.filterwarnings("ignore")

import logging
import sys
from datetime import datetime

import optuna

def setup_logging(log_file=None):
    if log_file is None:
        log_file = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file

def setup_optimizer(model, learning_rate, weight_decay, gcn_learning_rate_scale=1.0):

    param_groups = []


    gcn_params = model.get_gcn_parameters()
    if gcn_params:
        param_groups.append({
            'params': gcn_params,
            'lr': learning_rate * gcn_learning_rate_scale, 
            'weight_decay': weight_decay
        })


    other_params = model.get_other_parameters()
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': learning_rate,
            'weight_decay': weight_decay
        })

    optimizer = torch.optim.Adam(param_groups)
    return optimizer


def train_tagging(args):

    train_data, val_input, val_trg, test_input, test_trg, loc_size, id2loc, max_len, adj_graph = load_dataset(args, 'csv')

    pad_token_id = PAD_TOKEN  # pad token id is the same for target as well

    estimation_model = Transformer_tagging(
        model_path=os.path.join(args.llm_model_path, args.llm_model_class),
        model_class=args.llm_model_class,
        model_dimension=args.hidden_size,
        fourier_dimension=args.hidden_size,
        time_dimension=args.hidden_size,
        vocab_size=loc_size+TOTAL_SPE_TOKEN,
        number_of_heads=args.num_heads,
        number_of_layers=args.num_layers,
        number_cls=args.num_cls,
        dropout_probability=args.dropout,

        gcn_ckpt_path=args.gcn_ckpt_path,
        gcn_num_features=args.gcn_num_features,
        gcn_hid_dim=args.gcn_hid_dim,
        gcn_num_conv_layers=args.gcn_num_conv_layers,
        gcn_dropout=args.gcn_dropout,
        gcn_reconstruct=args.gcn_reconstruct,

        device=args.device
    ).to(args.device)


    A = calculate_laplacian_matrix(adj_graph, mat_type='hat_rw_normd_lap_mat')

    train_dataset = TrajectoryTaggingDataset(train_data, args, max_len, drop_num=[1,2,3,4], drop_ratio=[0.2,0.3,0.4,0.5,0.6], id2loc=id2loc)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataloader_collate)

    val_dataset = TestingTaggingDataset(val_input, val_trg, args, max_len)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,collate_fn=dataloader_collate_test)

    test_dataset = TestingTaggingDataset(test_input, test_trg, args, max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataloader_collate_test)
    cls_weight = torch.tensor([0.5, 1, 1, 1.5, 1.5], dtype=torch.float).to(args.device)

    ce_loss = nn.CrossEntropyLoss(reduction='none', weight=cls_weight)
    cl_loss = CL_Loss(args.temperature, args.device)
    A = torch.from_numpy(A).float().to_sparse().to(device=args.device)

    optimizer = torch.optim.Adam(estimation_model.parameters(), lr=args.lr)

    best_f1 = 0
    for epoch in range(args.num_epochs):
        # Training loop
        estimation_model.train()
        for iteration, (batch_data, batch_cl) in enumerate(train_dataloader):
            batch_data = tuple(t.to(args.device) for t in batch_data)
            batch_cl = tuple(t.to(args.device) for t in batch_cl)
            batch_enc_loc, batch_enc_time, batch_enc_coor, batch_lengths, batch_target = batch_data
            batch_cl_loc, batch_cl_time, batch_cl_coor, batch_cl_lengths = batch_cl

            src_mask, num_src_tokens = get_masks_and_count_tokens(batch_enc_loc, pad_token_id)
            attn_mask_cl, _ = get_masks_and_count_tokens_src(batch_cl_loc, pad_token_id)

            pred = estimation_model(batch_enc_loc, batch_enc_time, batch_enc_coor, src_mask, A, 'tagging')

            optimizer.zero_grad()

            if epoch >= args.warm_up_epochs:
                loss_tagging = loss_func(pred.view(-1, args.num_cls), batch_target.view(-1), src_mask.squeeze(), ce_loss)
                cl_outputs = estimation_model(batch_cl_loc, batch_cl_time, batch_cl_coor, attn_mask_cl, A, 'contrastive')
                loss_cl = cl_loss(estimation_model, cl_outputs, attn_mask_cl, batch_cl_lengths)

                loss = args.ce_weight * loss_tagging + args.cl_weight * loss_cl

                if iteration % 50 == 0:
                    logger.info("Epoch: {0}, Iteration: {1}\tLoss: {2:.4f}" \
                          .format(epoch, iteration, loss.item()))

            else:
                loss_tagging = loss_func(pred.view(-1, args.num_cls), batch_target.view(-1), src_mask.squeeze(), ce_loss)
                loss_cl = 0
                loss = args.ce_weight * loss_tagging + args.cl_weight * loss_cl

                if iteration % 50 == 0:
                    logger.info("Epoch: {0}, Iteration: {1}\tLoss: {2:.4f}" \
                          .format(epoch, iteration, loss.item()))

            loss.backward()
            optimizer.step()


        # Validation loop
        estimation_model.eval()

        val_preds, val_labels, val_lengths = validation(val_dataloader, estimation_model, A, args.device)
        assert len(val_preds) == len(val_labels)
        logger.info("length of preds: {}, evaluating validation set".format(len(val_labels)))
        prec, rec, f1_micro, f1_macro = evaluation_multiclass(val_preds, val_labels, val_lengths)

        if f1_micro + f1_macro > best_f1:
            torch.save(estimation_model.state_dict(), os.path.join(args.model_path, args.model_name))

            best_f1 = f1_micro + f1_macro

            if epoch >= args.test_epoch:
                test_preds, test_labels, test_lengths = validation(test_dataloader, estimation_model, A, args.device)
                assert len(test_preds) == len(test_trg)
                logger.info("best result so far, length of preds: {}, evaluating testset".format(len(test_preds)))
                prec, rec, f1_micro, f1_macro = evaluation_multiclass(test_preds, test_labels, test_lengths)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TrajectoryEnrichment')
    parser.add_argument("--dropout", type=float, default=0.1,
            help="dropout probability")
    parser.add_argument("--hidden_size", type=int, default=128,
            help="number of hidden dimension")
    parser.add_argument("--num_heads", type=int, default=4,
            help="number of heads")
    parser.add_argument("--out_size", type=int, default=128,
            help="number of output dim")
    parser.add_argument("--num_layers", type=int, default=4,
            help="number of encoder/decoder layers")
    parser.add_argument("--num_epochs", type=int, default=150,
            help="number of minimum training epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="number of batch size")
    parser.add_argument("--num_cls", type=int, default=5,
                        help="number of classes")
    parser.add_argument("--warm_up_epochs", type=int, default=10,
                        help="number of warm up epochs")
    parser.add_argument("--test_epoch", type=int, default=0,
                        help="perform testing after certain epochs")
    parser.add_argument("--cl_weight", type=float, default=0.2,
                        help="contrastive loss weight")
    parser.add_argument("--ce_weight", type=float, default=1.0,
                        help="location recovery loss weight")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="temperature parameter for contrastive loss")
    parser.add_argument("--gpu", type=int, default= 1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=0.0003,
            help="learning rate")

    parser.add_argument("--gcn_lr", type=float, default=0.001,
                        help="GCN learning rate")
    parser.add_argument("--gcn_wd", type=float, default=0.001,
                        help="GCN weight decay")
    parser.add_argument("--gcn_lrs", type=float, default=0.001,
                        help="GCN learning rate scale")


    parser.add_argument("--gcn_num_features", type=int, default=128,
                        help="GCN learning rate scale")
    parser.add_argument("--gcn_hid_dim", type=int, default=128,
                        help="GCN learning rate scale")
    parser.add_argument("--gcn_num_conv_layers", type=int, default=2,
                        help="GCN learning rate scale")
    parser.add_argument("--gcn_dropout", type=float, default=0.1,
                        help="GCN learning rate scale")
    parser.add_argument("--gcn_reconstruct", type=float, default=1.0,
                        help="GCN learning rate scale")

    parser.add_argument('--data_path', type=str, default='../data',
                        help='Dataset path')
    parser.add_argument('--data_name', type=str, default='chengdu',
                        help='Dataset name')
    parser.add_argument('--model_path', type=str, default='../model/train_chengdu_test_beijing',
                        help='Dataset path')
    parser.add_argument('--model_name', type=str, default='model_estimation',
                        help='model name')
    parser.add_argument('--llm_model_class', type=str, default='gpt2',
                        help='Dataset path')
    parser.add_argument('--llm_model_path', type=str, default='../params',
                        help='Dataset path')
    parser.add_argument("--gcn_ckpt_path", type=str, default='../storage/chengdu_train_beijing_test/chengdu,proto_pretrained_model.pt',
                        help="GCN learning rate scale")

    args = parser.parse_args()
    cuda_condition = torch.cuda.is_available() and args.gpu
    args.device = torch.device("cuda" if cuda_condition else "cpu")

    logger, log_file = setup_logging(os.path.join(args.model_path, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    logger.info(args)
    train_tagging(args)







