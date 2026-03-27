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

class Transformer_tagging(nn.Module):

    def __init__(self, model_path, model_class, model_dimension, fourier_dimension, time_dimension, vocab_size,
                 number_of_heads, number_of_layers, number_cls, dropout_probability,
                 gcn_ckpt_path, gcn_num_features, gcn_hid_dim, gcn_num_conv_layers, gcn_dropout, gcn_reconstruct,
                 device, log_attention_weights=False, position_encoding=True):
        super(Transformer_tagging, self).__init__()

        self.encoder, self.tokenizer, self.emb_size, self.hidden_size = get_encoder(model_path, model_class)


        self.src_embedding = nn.Embedding(vocab_size, self.emb_size)
        self.device = device

        self.model_class = model_class

        self.fourier_dimension = self.emb_size
        self.time_dimension = self.emb_size
        self.vocab_size = vocab_size


        self.time_embedding = LearnableFourierPositionalEncoding(1, self.fourier_dimension, self.time_dimension)
        self.dist_embedding = LearnableFourierPositionalEncoding(2, self.fourier_dimension, self.time_dimension)

        self.pos_encoding = position_encoding


        self.src_embedding_cxt = nn.Embedding(vocab_size, gcn_hid_dim)
        self.src_linear_cxt = nn.Linear(gcn_hid_dim, self.emb_size)
        self.gcn = PretrainedGCNAdapter(
            ckpt_path=gcn_ckpt_path,
            num_features=gcn_num_features,
            hid_dim=gcn_hid_dim,  
            num_conv_layers=gcn_num_conv_layers,  
            dropout=gcn_dropout,
            reconstruct=gcn_reconstruct,  
            map_location="cpu"  
        ).to(self.device)


        if position_encoding is True:
            self.pos_embedding = nn.Embedding(200, self.emb_size)



        self.mlp = nn.Linear(self.emb_size, number_cls)

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

    def forward(self, src_token_ids_batch, src_time_batch, src_coor_batch, src_mask, adj_graph, type):
        src_representations = self.encode(src_token_ids_batch, src_time_batch, src_coor_batch, src_mask, adj_graph)
        
        if type == 'tagging':
            outputs = self.decode(src_representations)
        elif type == 'contrastive':
            outputs = src_representations

        return outputs


    def encode(self, src_token_ids_batch, src_time_batch, src_dist_batch, src_mask, adj_graph):
        (bs, seq_len) = src_token_ids_batch.shape

        x = self.src_embedding_cxt(torch.arange(self.vocab_size, device=self.device))
        src_cxt_embeddings = self.gcn(x, adj_graph)
        src_cxt_embeddings = self.src_linear_cxt(src_cxt_embeddings)

        src_embeddings_batch = self.src_embedding(src_token_ids_batch)  # get embedding vectors for src token ids
        src_embeddings_batch_cxt = src_cxt_embeddings[src_token_ids_batch.view(-1)].view(bs, seq_len, -1)
        src_embeddings_batch = src_embeddings_batch_cxt + src_embeddings_batch

        if self.pos_encoding:
            src_pos_batch = torch.arange(src_token_ids_batch.size(1), device=self.device).unsqueeze(0).repeat(src_token_ids_batch.size(0), 1)
            src_pos_embeddings_batch = self.pos_embedding(src_pos_batch)
            src_embeddings_batch = src_embeddings_batch + src_pos_embeddings_batch

        src_time_embeddings_batch = self.time_embedding(src_time_batch)
        src_dist_embeddings_batch = self.dist_embedding(src_dist_batch)

        src_representations_batch = self.norm(
            src_embeddings_batch + src_time_embeddings_batch + src_dist_embeddings_batch
        )

        model_dtype = next(self.encoder.parameters()).dtype
        src_representations_batch = src_representations_batch.to(dtype=model_dtype)
        position_ids = torch.arange(0, src_representations_batch.shape[1], device=src_representations_batch.device).unsqueeze(0).repeat(
            src_representations_batch.shape[0], 1)

        if self.model_class == 'gpt2':
            src_representations_batch = self.encoder(inputs_embeds=src_representations_batch, attention_mask=src_mask.unsqueeze(-1), output_hidden_states=True).hidden_states[-1]

        return src_representations_batch

    def decode(self, src_representations):
        outputs = self.mlp(src_representations)
        return outputs


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
        # Step 1. Compute Fourier features (eq. 2)
        projected = self.Wr(x)
        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        F = 1 / np.sqrt(self.F_dim) * torch.cat([cosines, sines], dim=-1)
        # Step 2. Compute projected Fourier features (eq. 6)
        Y = self.mlp(F)
        # Step 3. Reshape to x's shape
        pos_enc = Y.reshape((B, N, self.D))
        return pos_enc


class CL_Loss(nn.Module):
    def __init__(self, temperature, device):
        super(CL_Loss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.cossim = nn.CosineSimilarity(dim=-1).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.type = 'mat'

    def forward(self, model, representations, mask, input_lengths):
        """
        contrastive learning loss given one pair sequences
        inputs: [batch1_data, batch2data], shape: 2B x S x D
        """
        traj_reps = torch.sum(representations * mask.squeeze().unsqueeze(-1).float(), dim=1) / input_lengths.unsqueeze(-1).float()  # 2B x D

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