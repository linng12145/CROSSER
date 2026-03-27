import numpy as np
import pandas as pd
import pickle
import argparse
import networkx as nx
import torch
import os
import sys
from pyproj import Transformer

sys.path.append('../')

from model import Transformer_insertion
from estimation_stage.model import Transformer_tagging
from utils import get_masks_and_count_tokens_src, get_masks_and_count_tokens_trg, calculate_laplacian_matrix
from dataloader import pad_arrays
from constants import *
from collections import defaultdict

import logging
import sys
from datetime import datetime

def setup_logging(log_file=None):
    if log_file is None:
        log_file = f"testing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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

def load_test_dataset(data_path, adj_path):
    test_path = os.path.join(data_path, 'traj_test.csv')
    lbs_test = pd.read_csv(test_path, converters={'trips_sparse': eval,
                                                  'num_labels': eval})

    if args.data_name == 'T-drive':
        id2loc = pickle.load(open(os.path.join(data_path, 'grid2center_Beijing.pickle'), 'rb'))
    elif args.data_name == 'chengdu':
        id2loc = pickle.load(open(os.path.join(data_path, 'grid2center_chengdu.pickle'), 'rb'))
    else:
        id2loc = pickle.load(open(os.path.join(data_path, 'grid2center_' + args.data_name + '.pickle'), 'rb'))

    logger.info("test data size {}, location num {}".format(len(lbs_test), len(id2loc)))

    to3414 = Transformer.from_crs("epsg:4326", "epsg:4575", always_xy=True)

    def dataset_collate(trips):
        trips_collate = []
        for trip in trips:
            trip_collate = []
            trip = trip.split(';')
            for loc in trip:
                idx, lon, lat, time = loc.split(',')
                trip_collate.append([int(idx), float(lon), float(lat), int(time)])
            trips_collate.append(trip_collate)
        return trips_collate


    def data_to_input(trips):
        trips_input = []
        for trip in trips:
            res = []
            time_min = trip[0][-1]
            for (loc, lon, lat, time) in trip:
                if loc == 'BLK':
                    res.append((BLK_TOKEN, 5000, 20447840.4, 4419792.3))
                else:
                    coords = id2loc[loc]
                    res.append((int(loc)+TOTAL_SPE_TOKEN, time-time_min, coords[0], coords[1]))
            trips_input.append(np.array(res))
        return trips_input


    def get_insertion_input_seq2seq(trips_drop, labels):
        res = []
        for trip_drop, label in zip(trips_drop, labels):
            assert len(trip_drop) == len(label)
            temp = []
            for loc, t in zip(trip_drop, label):
                if t == 0:
                    temp.append(loc)
                else:
                    temp.append(loc)
                    temp.append(['BLK', 'BLK', 'BLK', 'BLK'])

            assert len(temp) == len(trip_drop) + np.sum(np.array(label)!=0)
            res.append(temp)
        return res


    loc_size = len(id2loc)
    loc2id = {loc: id for id, loc in id2loc.items()}

    adj_pd = pd.read_csv(adj_path)
    adj_pd = adj_pd.add({'src': TOTAL_SPE_TOKEN, 'dst': TOTAL_SPE_TOKEN, 'weight': 0})
    G = nx.DiGraph()
    G.add_nodes_from(list(range(loc_size + TOTAL_SPE_TOKEN)))
    src, dst, weights = adj_pd['src'].values.tolist(), adj_pd['dst'].values.tolist(), adj_pd['weight'].values.tolist()
    G.add_weighted_edges_from(zip(src, dst, weights))
    adj_graph = nx.to_numpy_array(G)

    test_traj = lbs_test['trips_sparse'].values.tolist()
    test_tgt = dataset_collate(lbs_test['trips_new'].values.tolist())
    drop_ratios = lbs_test['drop_ratio'].values.tolist()

    max_len = 60

    logger.info("test num {}, target {}, " \
          .format(len(test_traj), len(test_tgt)))

    # + special tokens: PAD, BOS, EOS, NUL, BLK
    test_input = data_to_input(test_traj)
    # test_target = data_to_input(test_tgt)
    test_target = test_tgt

    return test_input, test_target, loc_size, id2loc, max_len, adj_graph, drop_ratios

def collate_multi_class_label(label):
    res = []
    if label == 0:
        res = []
    elif label == 1:
        res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(4)]
    elif label == 2:
        res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(9)]
    elif label == 3:
        res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(15)]
    elif label == 4:
        res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(25)]
    return res

def get_insertion_input_data(traj, num_label):
    res, masked_pos = [], []
    assert len(traj) == len(num_label)
    for (record, label) in zip(traj, num_label):
        if label == 0:
            res.append(record)
        else:
            res.append(record)
            blk_tokens = collate_multi_class_label(label)
            masked_pos.extend(list(range(len(res), len(res)+len(blk_tokens))))
            res.extend(blk_tokens)

    return np.array(res), np.array(masked_pos, dtype=np.int_)



def test_twostage(args):
    data_path = os.path.join(args.data_path, args.data_name)
    adj_path = os.path.join(data_path, 'graph_A.csv')
    estimation_model_path = os.path.join(args.model_path, 'model_estimation')
    recovery_model_path = os.path.join(args.model_path, 'model_recovery')


    test_input, test_target, loc_size, id2loc, max_len, adj_graph, drop_ratios = load_test_dataset(data_path, adj_path)
    tagging_model = Transformer_tagging(
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

    checkpoint = torch.load(estimation_model_path)

    model_dict = tagging_model.state_dict()

    pretrained_dict = {k: v for k, v in checkpoint.items()
                       if k in model_dict and model_dict[k].shape == v.shape}

    model_dict.update(pretrained_dict)
    tagging_model.load_state_dict(model_dict)

    insertion_model = Transformer_insertion(
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


    checkpoint = torch.load(recovery_model_path)

    model_dict = insertion_model.state_dict()

    pretrained_dict = {k: v for k, v in checkpoint.items()
                       if k in model_dict and model_dict[k].shape == v.shape}


    model_dict.update(pretrained_dict)
    insertion_model.load_state_dict(model_dict)

    A = calculate_laplacian_matrix(adj_graph, mat_type='hat_rw_normd_lap_mat')
    A = torch.from_numpy(A).float().to_sparse().to(device=args.device)

    ### Stage 1: tagging for BLK token
    test_size, eval_batch = len(test_input), args.batch_size
    num_iter = int(np.ceil(len(test_input) / eval_batch))
    logger.info("tagging stage: {}".format(num_iter))
    tagging_model.eval()
    tagging_preds = []
    for i in range(num_iter):
        if not i % 10:
            logger.info(f"estimation:{i}/{num_iter}")
        with torch.no_grad():
            traj_inp = test_input[i * eval_batch: min((i + 1) * eval_batch, test_size)]
            traj_inp = pad_arrays(traj_inp)
            # traj_inp = test_input[i]
            traj_inp_loc, traj_inp_time, traj_inp_coors = traj_inp[:, :, 0], traj_inp[:, :, 1:2], traj_inp[:, :, 2:]
            traj_inp_loc = torch.tensor(traj_inp_loc, dtype=torch.long, device=args.device)
            traj_inp_time = torch.tensor(traj_inp_time, dtype=torch.float, device=args.device)
            traj_inp_coors = torch.tensor(traj_inp_coors, dtype=torch.float, device=args.device)

            src_mask, _ = get_masks_and_count_tokens_src(traj_inp_loc, PAD_TOKEN)
            outputs = tagging_model(traj_inp_loc, traj_inp_time, traj_inp_coors, src_mask, A, 'tagging')
            pred = torch.argmax(outputs, dim=-1).cpu().numpy()
            tagging_preds.extend(pred)

    assert len(tagging_preds) == len(test_input)

    cnt = 0
    logger.info("processing tagging result to the input of stage 2 model")
    insertion_inputs, masked_pos = [], []
    for trip_drop, tag in zip(test_input, tagging_preds):
        tag = tag[:len(trip_drop)]
        insertion_input, masked = get_insertion_input_data(trip_drop, tag)

        insertion_inputs.append(insertion_input)
        masked_pos.append(masked)


    ### Stage 2: insertion for BLK tokens

    insertion_model.eval()
    inputs = []
    final_preds = []
    for i in range(num_iter):
        if not i % 10:
            logger.info(f"insertion:{i}/{num_iter}")
        with torch.no_grad():
            traj_inp = insertion_inputs[i * eval_batch: min((i + 1) * eval_batch, test_size)]
            masked_pos_np = masked_pos[i * eval_batch: min((i + 1) * eval_batch, test_size)]
            lengths = np.array(list(map(len, traj_inp)))
            masked_pos_lengths = np.array(list(map(len, masked_pos_np)))

            traj_inp = pad_arrays(traj_inp)
            traj_locs, traj_tms, traj_coors = traj_inp[:, :, 0], traj_inp[:, :, 1:2], traj_inp[:, :, 2:]

            traj_locs = torch.tensor(traj_locs, dtype=torch.long, device=args.device)
            traj_tms = torch.tensor(traj_tms, dtype=torch.float, device=args.device)
            traj_coors = torch.tensor(traj_coors, dtype=torch.float, device=args.device)

            inputs.extend(traj_locs.cpu().numpy())
            masked_pos_batch = pad_arrays(masked_pos_np)

            masked_pos_batch = torch.tensor(masked_pos_batch, dtype=torch.long, device=args.device)
            batch_pred_inputs = torch.tensor([BLK_TOKEN] * traj_locs.size(0), dtype=torch.long,
                                             device=args.device).unsqueeze(1)

            for idx in range(masked_pos_batch.shape[1]):
                attn_mask, _ = get_masks_and_count_tokens_trg(torch.cat([traj_locs, batch_pred_inputs], dim=1),
                                                              PAD_TOKEN)
                batch_masked_pos_cur = masked_pos_batch[:, :idx + 1]

                trg_probs = insertion_model(traj_locs, traj_tms, traj_coors, attn_mask, A, 'recovery', batch_masked_pos_cur,
                                            batch_pred_inputs)

                last_words_batch = trg_probs[:, idx]  # B x vocab_size
                pred_locs = torch.argmax(last_words_batch, dim=-1)

                batch_pred_inputs = torch.cat([batch_pred_inputs, pred_locs.unsqueeze(1)], dim=1)

            output_pred_locs = batch_pred_inputs[:, 1:].cpu().numpy()  # remove the first blk token
            output_locs = traj_locs.cpu().numpy()
            batch_preds_post = []
            for idx, (pred, masked_p, length, masked_pos_length) in enumerate(
                    zip(output_pred_locs, masked_pos_np, lengths, masked_pos_lengths)):
                masked_p = masked_p[:masked_pos_length]
                output_locs[idx, masked_p] = pred[:masked_pos_length]
                batch_preds_post.append(output_locs[idx, :length])

        final_preds.extend(batch_preds_post)
    # logger.info(len(final_preds), len(test_input))
    logger.info("{}, {}".format(len(final_preds), len(test_input)))
    assert len(final_preds) == len(test_input)
    # logger.info("length of preds: {}, evaluating".format(len(final_preds)))
    logger.info("Length of final_preds: %s, Length of test_input: %s", len(final_preds), len(test_input))

    prec, rec, recovery, m_prec = evaluate(insertion_inputs, final_preds, test_target, id2loc, max_len)


def evaluate(test_input, preds, test_target, id2loc, maxlen):
    # to3414 = Transformer.from_crs("epsg:4326", "epsg:3414", always_xy=True)
    recall_total, precision_total, recovery_total, micro_precision_total = [], [], [], []

    for idx, (drop, pred, label) in enumerate(zip(test_input, preds, test_target)):
        label = [l[0] for l in label]
        pred = [p - TOTAL_SPE_TOKEN for p in pred if p >= TOTAL_SPE_TOKEN]


        recall = len(set(pred).intersection(set(label))) / len(label)
        precision = len(set(pred).intersection(set(label))) / len(pred)

        drop = [p[0] - TOTAL_SPE_TOKEN for p in drop if p[0] >= TOTAL_SPE_TOKEN]

        expected = set(label) - set(drop)
        if len(expected) > 0:
            recovery = len(set(pred).intersection(expected)) / len(expected)
        else:
            recovery = 1

        pred_missing = [loc for loc in pred if loc not in drop]
        if len(pred_missing) > 0:
            micro_prec = len(set(pred_missing).intersection(expected)) / len(pred_missing)
        else:
            micro_prec = 0

        recall_total.append(recall)
        recovery_total.append(recovery)
        precision_total.append(precision)
        micro_precision_total.append(micro_prec)
        # hauss_dist_total.append(hauss)

    logger.info("average recall {}, average precision {}, average micro-recall {}, average micro-precision {}". \
          format(np.mean(recall_total), np.mean(precision_total), np.mean(recovery_total),
                 np.mean(micro_precision_total)))
    prec, recall, recovery, m_prec = np.mean(precision_total), np.mean(recall_total), np.mean(
        recovery_total), np.mean(micro_precision_total)

    return prec, recall, recovery, m_prec



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TrajectoryEnrichment')
    parser.add_argument("--dropout", type=float, default=0.2,
            help="dropout probability")
    parser.add_argument("--hidden_size", type=int, default=128,
            help="number of hidden dimension")
    parser.add_argument("--num_heads", type=int, default=4,
            help="number of heads")
    parser.add_argument("--out_size", type=int, default=128,
            help="number of output dim")
    parser.add_argument("--num_layers", type=int, default=4,
            help="number of encoder/decoder layers")
    parser.add_argument("--num_epochs", type=int, default=50,
            help="number of minimum training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="number of batch size")
    parser.add_argument("--num_cls", type=int, default=5,
                        help="number of classes")
    parser.add_argument("--num", type=int, default=1,
                        help="number of dropped segments for each trip")
    parser.add_argument("--rate", type=float, default=0.4,
                        help="dropping rate for each segment")
    parser.add_argument("--gpu", type=int, default= 1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=0.00005,
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

    parser.add_argument("--num_warmup_steps", type=int, default=3000,
                        help="number of warm-up steps")
    parser.add_argument('--model_path', type=str, default='../model/train_chengdu_test_beijing',
                        help='Model path')
    parser.add_argument('--data_path', type=str, default='../data',
                        help='Dataset path')
    parser.add_argument('--data_name', type=str, default='T-drive',
                        help='Dataset name')
    parser.add_argument('--llm_model_class', type=str, default='gpt2',
                        help='Dataset path')
    parser.add_argument('--llm_model_path', type=str, default='../params',
                        help='Dataset path')
    parser.add_argument("--gcn_ckpt_path", type=str, default='../storage/chengdu_train_beijing_test/chengdu,proto_pretrained_model.pt',
                        help="GCN learning rate scale")


    args = parser.parse_args()
    cuda_condition = torch.cuda.is_available() and args.gpu
    args.device = torch.device("cuda" if cuda_condition else "cpu")
    # args.data_path = os.path.join(args.dataset, args.data_path)
    args.sample = False

    logger, log_file = setup_logging(os.path.join(args.model_path, f"testing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    logger.info(args)
    test_twostage(args)

