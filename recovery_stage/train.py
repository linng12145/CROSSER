import numpy as np
import pandas as pd
import pickle
import random
import argparse
import os
import torch
import torch.nn as nn
from model import Transformer_insertion, CL_Loss
from pyproj import Transformer

from dataloader import TrajectoryInfillingDataset, TestingInfillingDataset, dataloader_collate, dataloader_collate_test
from torch.utils.data import DataLoader
from utils import *
from constants import *

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

def train_recovery(args, trial=None):

    train_data, val_input, val_num_labels, val_trg, test_input, test_num_labels, test_trg, loc_size, id2loc, max_len, adj_graph = load_dataset(args, 'csv')

    pad_token_id = PAD_TOKEN  # pad token id is the same for target as well

    recovery_model = Transformer_insertion(
        model_path=os.path.join(args.llm_model_path, args.llm_model_class),
        model_class=args.llm_model_class,
        model_dimension=args.hidden_size,
        fourier_dimension=args.hidden_size,
        time_dimension=args.hidden_size,
        src_vocab_size=loc_size+TOTAL_SPE_TOKEN,
        trg_vocab_size=loc_size+TOTAL_SPE_TOKEN,
        number_of_heads=args.num_heads,
        number_of_layers=args.num_layers,
        dropout_probability=args.dropout,

        gcn_ckpt_path=args.gcn_ckpt_path,
        gcn_num_features=args.gcn_num_features,
        gcn_hid_dim=args.gcn_hid_dim,
        gcn_num_conv_layers=args.gcn_num_conv_layers,
        gcn_dropout=args.gcn_dropout,
        gcn_reconstruct=args.gcn_reconstruct,

        max_len=max_len,
        device = args.device
    ).to(args.device)

    A = calculate_laplacian_matrix(adj_graph, mat_type='hat_rw_normd_lap_mat')

    train_dataset = TrajectoryInfillingDataset(train_data, args, max_len, drop_num=[1, 2, 3, 4],
                                             drop_ratio=[0.2, 0.3, 0.4, 0.5, 0.6], id2loc=id2loc)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataloader_collate)

    val_dataset = TestingInfillingDataset(val_input, val_num_labels, val_trg, args, max_len)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataloader_collate_test)

    test_dataset = TestingInfillingDataset(test_input, test_num_labels, test_trg, args, max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataloader_collate_test)


    ce_loss = nn.CrossEntropyLoss(reduction='none')
    cl_loss = CL_Loss(args.temperature, args.device)

    # optimizer = torch.optim.Adam(recovery_model.parameters(), lr=args.lr)
    optimizer = setup_optimizer(recovery_model, learning_rate=args.gcn_lr, weight_decay=args.gcn_wd, gcn_learning_rate_scale=args.gcn_lrs)
    A = torch.from_numpy(A).float().to_sparse().to(device=args.device)


    best_rec = 0
    for epoch in range(args.num_epochs):
        # Training loop
        recovery_model.train()
        for iteration, (batch_data, batch_cl) in enumerate(train_dataloader):
            batch_data = tuple(t.to(args.device) for t in batch_data)
            batch_cl = tuple(t.to(args.device) for t in batch_cl)
            batch_loc, batch_time, batch_coor, batch_lengths, batch_masked_pos, batch_pred_inputs, batch_pred_targets, batch_masked_weight = batch_data
            batch_cl_seqs, batch_cl_tms, batch_cl_coors, batch_cl_lengths = batch_cl

            attn_mask, num_src_tokens = get_masks_and_count_tokens_trg(torch.cat([batch_loc, batch_pred_inputs], dim=1), pad_token_id)
            attn_mask_cl, _ = get_masks_and_count_tokens_trg(batch_cl_seqs, pad_token_id)

            pred = recovery_model(batch_loc, batch_time, batch_coor, attn_mask, A, 'recovery', batch_masked_pos, batch_pred_inputs)

            optimizer.zero_grad()

            if epoch >= args.warm_up_epochs:
                loss_mask = ce_loss(pred.view(-1, loc_size+TOTAL_SPE_TOKEN), batch_pred_targets.flatten())
                loss_rec = (loss_mask * batch_masked_weight.flatten()).sum()/ batch_masked_weight.sum()

                cl_output_representations = recovery_model(batch_cl_seqs, batch_cl_tms, batch_cl_coors, attn_mask_cl, A, 'contrastive')
                loss_cl = cl_loss(recovery_model, cl_output_representations, batch_cl_lengths)

                loss = args.ce_weight * loss_rec + args.cl_weight * loss_cl

                if iteration % 50 == 0:
                    logger.info("Epoch: {0}, Iteration: {1}\tLoss: {2:.4f}" \
                          .format(epoch, iteration, loss.item()))

            else:
                loss_mask = ce_loss(pred.view(-1, loc_size + TOTAL_SPE_TOKEN), batch_pred_targets.flatten())
                loss_rec = (loss_mask * batch_masked_weight.flatten()).sum() / batch_masked_weight.sum()
                loss = args.ce_weight * loss_rec

                if iteration % 50 == 0:
                    logger.info("Epoch: {0}, Iteration: {1}\tLoss: {2:.4f}" \
                          .format(epoch, iteration, loss.item()))

            loss.backward()
            optimizer.step()

        if epoch % 3 == 0:

            # Validation loop
            recovery_model.eval()

            val_preds = validation(val_dataloader, recovery_model, A, args.device)
            assert len(val_preds) == len(val_trg)
            logger.info("length of preds: {}, evaluating validation set".format(len(val_preds)))
            prec, rec, recovery, micro_prec = evaluation(val_input, val_preds, val_trg, id2loc, max_len)

            if trial is not None:
                trial.report(rec, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            if rec > best_rec:
                torch.save(recovery_model.state_dict(), os.path.join(args.model_path, args.model_name))
                best_rec = rec

                if epoch >= args.test_epoch:
                    preds = validation(test_dataloader, recovery_model, A, args.device)
                    assert len(preds) == len(test_trg)
                    logger.info("best result so far, length of preds: {}, evaluating testset".format(len(preds)))
                    prec, rec, recovery, micro_prec = evaluation(test_input, preds, test_trg, id2loc, max_len)

    return best_rec



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
    parser.add_argument("--batch_size", type=int, default=64,
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
    parser.add_argument("--lr", type=float, default=0.0002,
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
    parser.add_argument('--llm_model_class', type=str, default='gpt2',
                        help='LLM class')
    parser.add_argument('--llm_model_path', type=str, default='../params',
                        help='LLM path')
    parser.add_argument('--model_path', type=str, default='../model/train_chengdu_test_beijing',
                        help='model path')
    parser.add_argument('--model_name', type=str, default='model_recovery',
                        help='model name')
    parser.add_argument("--gcn_ckpt_path", type=str, default='../storage/chengdu_train_beijing_test/chengdu,proto_pretrained_model.pt',
                        help="GCN learning rate scale")

    args = parser.parse_args()
    cuda_condition = torch.cuda.is_available() and args.gpu
    args.device = torch.device("cuda" if cuda_condition else "cpu")

    logger, log_file = setup_logging(os.path.join(args.model_path, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    logger.info(args)
    train_recovery(args)

