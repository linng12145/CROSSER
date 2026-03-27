import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from peft import LoraModel, LoraConfig

import sys
sys.path.append('../')
from utils import *
from phi_model import PhiModel
from pretrain_GCN.load_pretrain import PretrainedGCNAdapter


class Transformer_insertion(nn.Module):

    def __init__(self, model_path, model_class, model_dimension, fourier_dimension, time_dimension, src_vocab_size, trg_vocab_size, number_of_heads, number_of_layers,
                 dropout_probability, gcn_ckpt_path, gcn_num_features, gcn_hid_dim, gcn_num_conv_layers, gcn_dropout, gcn_reconstruct,
                 max_len, device, max_input_length=200, log_attention_weights=False, learnable_pos=True):
        super().__init__()

        self.encoder, self.tokenizer, self.emb_size, self.hidden_size = get_encoder(model_path, model_class)

        self.learnable_pos = learnable_pos
        self.device = device

        self.model_class = model_class

        self.fourier_dimension = self.emb_size
        self.time_dimension = self.emb_size
        self.src_vocab_size = src_vocab_size



        self.src_embedding_cxt = nn.Embedding(src_vocab_size, gcn_num_features)
        self.src_linear_cxt = nn.Linear(gcn_hid_dim, self.emb_size)
        self.src_embedding_loc = nn.Embedding(src_vocab_size, self.emb_size)

        self.gcn = PretrainedGCNAdapter(
            ckpt_path=gcn_ckpt_path,
            num_features=gcn_num_features,
            hid_dim=gcn_hid_dim, 
            num_conv_layers=gcn_num_conv_layers,  
            dropout=gcn_dropout,
            reconstruct=gcn_reconstruct, 
            map_location="cpu"  
        ).to(self.device)

        if learnable_pos:
            self.src_pos_embedding = Embedding(max_input_length, self.emb_size)
            self.time_embedding = LearnableFourierPositionalEncoding(1, self.fourier_dimension, self.time_dimension)
            self.dist_embedding = LearnableFourierPositionalEncoding(2, self.fourier_dimension, self.time_dimension)
        else:
            self.src_pos_embedding = PositionalEncoding(self.emb_size, dropout_probability, device)


        self.decoder = nn.Sequential(nn.Linear(self.emb_size, self.emb_size),
                                     nn.ReLU(),
                                     nn.Linear(self.emb_size, trg_vocab_size))

        self.projection = nn.Sequential(nn.Linear(self.emb_size, self.emb_size),
                                        nn.ReLU(),
                                        nn.Linear(self.emb_size, 512))

        self.norm = nn.LayerNorm(self.emb_size)

        self.init_params()

    def init_params(self, default_initialization=False):
        if not default_initialization:
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def get_gcn_parameters(self):
        
        return [p for p in self.gcn.parameters() if p.requires_grad]

    def get_other_parameters(self):
        
        gcn_params = set(self.gcn.parameters())
        return [p for p in self.parameters() if p.requires_grad and p not in gcn_params]

    def forward(self, src_token_ids_batch, src_time_batch, src_dist_batch, attn_mask, adj_graph, type,
                masked_pos=None, src_pred_inputs_batch=None):
        src_representations_batch = self.encode(src_token_ids_batch, src_time_batch, src_dist_batch, attn_mask, adj_graph, type,
                                                src_pred_inputs_batch, masked_pos)
        if type == 'recovery':
            outputs = self.decode(src_representations_batch, src_token_ids_batch.size(1))
        elif type == 'contrastive':
            outputs = src_representations_batch

        return outputs


    def encode(self, src_token_ids_batch, src_time_batch, src_dist_batch, attn_mask, adj_graph, type,
               pred_inputs_batch=None, masked_pos=None):
        (bs, seq_len) = src_token_ids_batch.shape
        if type == 'recovery':
            src_token_inputs_batch = torch.cat([src_token_ids_batch, pred_inputs_batch], dim=1)
            times_batch = torch.cat([src_time_batch, torch.ones(bs, pred_inputs_batch.shape[1], src_time_batch.shape[2],
                                                                dtype=torch.float, device=self.device)*PAD_TIME], dim=1)
            dists_pad = torch.tensor([[PAD_LON, PAD_LAT]], device=self.device, dtype=torch.float).unsqueeze(1).repeat(bs, pred_inputs_batch.shape[1], 1)
            dists_batch = torch.cat([src_dist_batch, dists_pad], dim=1)
        elif type == 'contrastive':
            src_token_inputs_batch = src_token_ids_batch
            times_batch = src_time_batch
            dists_batch = src_dist_batch

        x = self.src_embedding_cxt(torch.arange(self.src_vocab_size, device=self.device))
        loc_cxt_embeddings = self.gcn(x, adj_graph)
        loc_cxt_embeddings = self.src_linear_cxt(loc_cxt_embeddings)

        token_embeddings_batch_cxt = loc_cxt_embeddings[src_token_inputs_batch.view(-1)].view(bs, src_token_inputs_batch.shape[1], -1)
        token_embeddings_batch_loc = self.src_embedding_loc(src_token_inputs_batch)  # get embedding vectors for src token ids
        token_embeddings_batch = token_embeddings_batch_cxt + token_embeddings_batch_loc

        if self.learnable_pos:
            src_pos_batch = torch.arange(src_token_ids_batch.size(1), device=self.device).unsqueeze(0).repeat(bs, 1)
            if masked_pos is None:
                pos_ids_batch = src_pos_batch
            else:
                pos_ids_batch = torch.cat([src_pos_batch, masked_pos[:, :1], masked_pos[:, :-1]], dim=1)
            pos_embeddings_batch = self.src_pos_embedding(pos_ids_batch)
            time_embeddings_batch = self.time_embedding(times_batch)
            dist_embeddings_batch = self.dist_embedding(dists_batch)
            # src_embeddings_batch = src_embeddings_batch + src_pos_embeddings_batch + src_time_embeddings_batch + src_dist_embeddings_batch
            embeddings_batch = token_embeddings_batch + pos_embeddings_batch + time_embeddings_batch + dist_embeddings_batch
        else:
            embeddings_batch = self.src_pos_embedding(token_embeddings_batch)  # add positional embedding


        if self.model_class == 'gpt2':
            src_representations_batch = self.encoder(inputs_embeds=embeddings_batch, attention_mask=attn_mask.unsqueeze(-1), output_hidden_states=True).hidden_states[-1]

        return src_representations_batch


    def decode(self, src_representations_batch, src_length):
        masked_output_probs= self.decoder(src_representations_batch[:, src_length:]) # B x Masked_S x H
        return masked_output_probs


#
# Input modules
#

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GCN, self).__init__()

        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)

        return x


class Embedding(nn.Module):

    def __init__(self, vocab_size, model_dimension):
        super().__init__()
        self.embeddings_table = nn.Embedding(vocab_size, model_dimension)
        self.model_dimension = model_dimension

    def forward(self, token_ids_batch):
        assert token_ids_batch.ndim == 2, f'Expected: (batch size, max token sequence length), got {token_ids_batch.shape}'

        embeddings = self.embeddings_table(token_ids_batch)

        return embeddings * math.sqrt(self.model_dimension)


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, F_dim: int, D: int, gamma=1.0):
        """
        :param M: each point has a M-dimensional positional values
        :param F_dim: depth of the Fourier feature dimension
        :param H_dim: hidden layer dimension
        :param D: positional encoding dimension
        :param gamma: parameter to initialize Wr
        """
        super().__init__()
        self.M = M
        self.F_dim = F_dim
        self.H_dim = D
        self.D = D
        self.gamma = gamma

        # Projection matrix on learned lines (used in eq. 2)
        self.Wr = nn.Linear(self.M, self.F_dim // 2, bias=False)
        # MLP (GeLU(F @ W1 + B1) @ W2 + B2 (eq. 6)
        self.mlp = nn.Sequential(
            nn.Linear(self.F_dim, self.H_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.H_dim, self.D)
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x):
        """
        Produce positional encodings from x
        :param x: tensor of shape [B, N, M] that represents N positions where each position is in the shape of M with batch size B,
        :return: positional encoding for X [B, N, D]
        """
        B, N, M = x.shape

        projected = self.Wr(x)
        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        F = 1 / np.sqrt(self.F_dim) * torch.cat([cosines, sines], dim=-1)
        Y = self.mlp(F)
        pos_enc = Y.reshape((B, N, self.D))
        return pos_enc


class PositionalEncoding(nn.Module):

    def __init__(self, model_dimension, dropout_probability, device, expected_max_sequence_length=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_probability)

        position_id = torch.arange(0, expected_max_sequence_length).unsqueeze(1)
        frequencies = torch.pow(10000., -torch.arange(0, model_dimension, 2, dtype=torch.float) / model_dimension)

        positional_encodings_table = torch.zeros(expected_max_sequence_length, model_dimension).to(device)
        positional_encodings_table[:, 0::2] = torch.sin(position_id * frequencies).to(device)  # sine on even positions
        positional_encodings_table[:, 1::2] = torch.cos(position_id * frequencies).to(device)  # cosine on odd positions

        self.register_buffer('positional_encodings_table', positional_encodings_table)

    def forward(self, embeddings_batch):
        assert embeddings_batch.ndim == 3 and embeddings_batch.shape[-1] == self.positional_encodings_table.shape[1], \
            f'Expected (batch size, max token sequence length, model dimension) got {embeddings_batch.shape}'

        positional_encodings = self.positional_encodings_table[:embeddings_batch.shape[1]]

        return self.dropout(embeddings_batch + positional_encodings)



class CL_Loss(nn.Module):
    def __init__(self, temperature, device):
        super(CL_Loss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.cossim = nn.CosineSimilarity(dim=-1).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.type = 'cos'

    def forward(self, model, representations, input_lengths):
        """
        contrastive learning loss given one pair sequences
        inputs: [batch1_data, batch2data], shape: 2B x S x D
        """
        # traj_reps = torch.sum(representations * mask.squeeze().unsqueeze(-1).float(), dim=1) / input_lengths.unsqueeze(-1).float()  # 2B x D

        idxs = (input_lengths-1).unsqueeze(-1).repeat(1, representations.size(-1)).unsqueeze(1) # 2B x 1 x D
        traj_reps = torch.gather(representations, 1, idxs).squeeze(1)
        traj_projs = model.projection(traj_reps)


        batch_size = traj_projs.shape[0] // 2
        batch_sample_one, batch_sample_two = torch.split(traj_projs, batch_size)

        if self.type == 'cos':
            sim11 = self.cossim(batch_sample_one.unsqueeze(1), batch_sample_one.unsqueeze(0)) / self.temperature
            sim22 = self.cossim(batch_sample_two.unsqueeze(1), batch_sample_two.unsqueeze(0)) / self.temperature
            sim12 = self.cossim(batch_sample_one.unsqueeze(1), batch_sample_two.unsqueeze(0)) / self.temperature
        else:
            sim11 = torch.matmul(batch_sample_one, batch_sample_one.T) / self.temperature
            sim22 = torch.matmul(batch_sample_two, batch_sample_two.T) / self.temperature
            sim12 = torch.matmul(batch_sample_one, batch_sample_two.T) / self.temperature


        mask_sim = torch.eye(sim12.shape[-1], dtype=torch.long).to(self.device)
        sim11[mask_sim == 1] = float("-inf")
        sim22[mask_sim == 1] = float("-inf")

        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * sim12.shape[-1], dtype=torch.long, device=logits.device)
        ce_loss = self.criterion(logits, labels)
        return ce_loss

#
# LLM Encoder architecture
#
def get_encoder(model_path, model_class):
    if model_class == 'gpt2':
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,  # Lora attention dimension.
            lora_alpha=32,  # The alpha parameter for Lora scaling.
            target_modules=["c_attn"],  # The names of the modules to apply Lora to.
            lora_dropout=0.02,  # The dropout probability for Lora layers.
        )
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        emb_size = model.config.n_embd
        hidden_size = model.config.n_embd

    else:
        raise NotImplementedError("model_class should be one of ['gpt2']")

    return LoraModel(model, lora_config, model_class), tokenizer, emb_size, hidden_size


class CustomPhiModel(PhiModel):
    """ Phi for traj modeling """

    _keys_to_ignore_on_load_missing = ["ladder_side_nets", "up_net"]

    # _keys_to_ignore_on_load_unexpected = [r"transformer\.h\.\d+\.mlp.(fc_in|fc_out)\.(weight|bias)"]

    def __init__(self, config, r=32):
        super().__init__(config)

        assert config.n_embd % r == 0, f"n_embd should be divisible by r, got {config.n_embd} and {r}"
        side_dim = config.n_embd // r

        self.side_dim = side_dim
        self.ladder_side_nets = nn.ModuleList([nn.Linear(config.n_embd, side_dim) for _ in range(config.n_layer)])
        self.up_net = nn.Linear(side_dim, config.n_embd)

    def forward(self, input_ids=None, inputs_embeds=None, past_key_values=None, attention_mask=None):
        if inputs_embeds is None:
            assert input_ids is not None, "You have to specify either input_ids or inputs_embeds"
            hidden_states = self.embd(input_ids)
        else:
            hidden_states = inputs_embeds

        for i, layer in enumerate(self.h):
            hidden_states = layer(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
            )

        # hidden_states = self.up_net(side_states) + hidden_states
        return hidden_states

class CustomLlamaModel(nn.Module):
    """ Phi for traj modeling """

    def __init__(self, model_path):
        super().__init__()

        self.model = AutoModelForCausalLM.from_pretrained(model_path).model
        self.embd = self.model.embed_tokens

    def forward(self, input_ids=None, inputs_embeds=None, past_key_values=None, attention_mask=None):
        past_key_values_length = 0

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(0, inputs_embeds.shape[1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = ()

        for decoder_layer in self.model.layers:
            all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,

            )

            hidden_states = layer_outputs[0]

        hidden_states = self.model.norm(hidden_states)
        # add hidden states from the last decoder layer
        all_hidden_states += (hidden_states,)
        return all_hidden_states


class CustomPythiaModel(nn.Module):
    """ Phi for traj modeling """

    def __init__(self, model_path):
        super().__init__()

        self.model = AutoModelForCausalLM.from_pretrained(model_path).gpt_neox
        self.embd = self.model.embed_in

    def forward(self, input_ids=None, inputs_embeds=None, past_key_values=None, attention_mask=None):
        past_key_values_length = 0

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(0, inputs_embeds.shape[1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = ()
        past_length = 0
        past_key_values = tuple([None] * self.model.config.num_hidden_layers)

        for i, (layer, layer_past) in enumerate(zip(self.model.layers, past_key_values)):
            all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,

                layer_past=layer_past,

            )
            hidden_states = outputs[0]

        hidden_states = self.model.final_layer_norm(hidden_states)
        # Add last hidden state

        all_hidden_states = all_hidden_states + (hidden_states,)
        return all_hidden_states