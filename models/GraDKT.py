from math import sqrt

import torch
from torch import nn
from torch.nn import (Embedding, Sequential, Linear, Sigmoid, ReLU, Dropout, LayerNorm, Softplus, CosineSimilarity)
from torch.nn.functional import (one_hot, softmax)
from torch_geometric.nn import GCN
from torch_geometric.utils import to_edge_index, add_remaining_self_loops

# GGN-Diffusion 模块
from torch_geometric.nn import MessagePassing
import torch_geometric

class GaussianDiffusionEncoder(nn.Module):
    def __init__(self, dim_q, dim_c, dim_g, dropout=0.1):
        super().__init__()
        self.encoder_mu = Sequential(
            Linear(dim_q + dim_c, dim_g), ReLU(), Dropout(dropout),
            Linear(dim_g, dim_g))
        self.encoder_logvar = Sequential(
            Linear(dim_q + dim_c, dim_g), ReLU(), Dropout(dropout),
            Linear(dim_g, dim_g))

    def forward(self, ht, ci):
        x = torch.cat([ht, ci], dim=-1)
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # reparameterization

class GaussianPropagation(MessagePassing):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__(aggr='add')
        self.lin = Linear(in_channels, out_channels)
        self.act = ReLU()
        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.lin(x)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        return self.dropout(self.act(aggr_out))
# 添加自门控模块（Self-Gating）
class SelfGating(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = Linear(dim, dim)
    
    def forward(self, x):
        gate = torch.sigmoid(self.linear(x))
        return x * gate
#扰动注入器 (Score Perturbator)
class ScorePerturbator(nn.Module):
    def __init__(self, dropout=0.2, noise_std=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.noise_std = noise_std
    def forward(self, score_embed):
        noise = torch.randn_like(score_embed) * self.noise_std
        return self.dropout(score_embed) + noise
#重建器 (Score Reconstructor)
class ScoreReconstructor(nn.Module):
    def __init__(self, dim_q):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(dim_q, dim_q),
            nn.ReLU(),
            nn.Linear(dim_q, 1),
            nn.Sigmoid()
        )

    def forward(self, dt_hat):
        return self.decoder(dt_hat).squeeze(-1)

class GraDKT(nn.Module):
    def __init__(self, num_c, num_q, num_o, dim_c, dim_q, dim_g,
                 num_heads, layer_g, top_k, lamb, map, option_list, dropout, bias):
        super().__init__()
        self.name = "GraDKT"

        ### Config
        self.num_c = num_c
        self.num_q = num_q
        self.num_o = num_o
        self.dim_c = dim_c
        self.dim_q = dim_q
        self.dim_g = dim_g
        self.top_k = top_k
        self.lamb = lamb
        self.concept_map = nn.Parameter(map, requires_grad=False)
        self.num_edge = to_edge_index(self.concept_map.to_sparse())[0].shape[1]
        self.option_mask = nn.Parameter(torch.zeros((self.num_q + 1, self.num_o), dtype=torch.int),
                                        requires_grad=False)
        self.score_perturb = ScorePerturbator(dropout=0.2, noise_std=0.1)
        self.score_recon = ScoreReconstructor(self.dim_q)

        for q, max_option in enumerate(option_list):
            self.option_mask[q, :max_option] = 1
        self.option_mask[-1, :2] = 1  # padding for chosen and valid unchosen response
        self.option_range = nn.Parameter(torch.tensor([[i for i in range(self.num_o)] for _ in range(self.num_q + 1)]),
                                         requires_grad=False)

        # Embedding
        self.concept_emb = Embedding(self.num_c, self.dim_c)
        self.question_emb = Embedding(self.num_q + 1, self.dim_q)
        self.response_emb = Embedding(self.num_q * self.num_o + 2, self.dim_q, padding_idx=-1)

        # 4.A Disentangled Response Encoder
        dim_h1 = self.dim_q // 2
        self.enc_correct = Sequential(
            Linear(self.dim_q, dim_h1), ReLU(), Dropout(dropout),
            Linear(dim_h1, self.dim_q))
        self.enc_wrong = Sequential(
            Linear(self.dim_q, dim_h1), ReLU(), Dropout(dropout),
            Linear(dim_h1, self.dim_q))
        self.enc_unchosen = Sequential(
            Linear(self.dim_q, dim_h1), ReLU(), Dropout(dropout),
            Linear(dim_h1, self.dim_q))
        self.attn_response = GraDKTLayer(self.dim_q, self.dim_q, num_heads, dropout, kq_same=True)

        # 4,B Knowledge Retriever
        self.attn_question = GraDKTLayer(self.dim_q, self.dim_q, num_heads, dropout, kq_same=True)
        self.attn_state = GraDKTLayer(self.dim_q, self.dim_q, num_heads, dropout, kq_same=True)

        # 4.C Concept Map Encoder
        dim_h2 = (self.dim_q + self.dim_c) // 2
        self.enc_concept = Sequential(
            Linear(self.dim_q + self.dim_c, dim_h2), ReLU(), Dropout(dropout),
            Linear(dim_h2, self.dim_g))
        dim_h3 = (self.dim_q + self.dim_c * 2) // 4
        self.enc_intensity = Sequential(
            Linear(self.dim_q + self.dim_c * 2, dim_h3), ReLU(), Dropout(dropout),
            Linear(dim_h3, 1))
        self.diff_encoder = GaussianDiffusionEncoder(self.dim_q, self.dim_c, self.dim_g, dropout)
        self.ggn = GaussianPropagation(self.dim_g, 1, dropout)

        # 4.D IRT-based Prediction
        self.w_relevance = Linear(self.dim_q, self.dim_c)
        self.mlp_diff = Sequential(
            Linear(self.dim_q, dim_h1), ReLU(), Dropout(dropout),
            Linear(dim_h1, 1))
        self.sigmoid = Sigmoid()
        self.relu = ReLU()
        self.softplus = Softplus()

        self.cossim = CosineSimilarity(dim=-1)
        self.temp = 0.05
        self.state_gru = nn.GRU(input_size=self.dim_q, hidden_size=self.dim_q, batch_first=True)
# 添加自门控模块，改善 AUC 表现
        self.gate_dt = SelfGating(self.dim_q)  # 用于解耦响应表示的门控
        self.gate_ht = SelfGating(self.dim_q)  # 用于知识状态表示的门控
        self.gate_mt = SelfGating(self.dim_g)  # 用于概念掌握表示的门控
    def forward(self, question, concept, score, option, unchosen,
                pos_score=None, pos_option=None, neg_score=None, neg_option=None, inference=False):
        self.batch_size, self.seq_len = question.shape

        seq_mask = torch.ne(score, -1)[:, :-1]
        qt_idx, c_qt_idx, ot_idx, ut_idx, ut_mask = self.get_index(question, concept, score, option, unchosen)

        # get question embedding
        qt = self.question_emb(qt_idx)  # (batch, seq, dim_q)

        # 4.A Response-State Fusion Encoder
        dt_hat = self.fuse_response_state(ot_idx, score, ut_idx, ut_mask)
        dt_hat = self.gate_dt(dt_hat)
        dt_hat_perturbed = self.score_perturb(dt_hat)    # 新增扰动
        score_recon_pred = self.score_recon(dt_hat_perturbed)  # 新增重建
        
        # 4.B Memory-Augmented Retriever
        qt_hat, ht = self.memory_fused_state(qt, dt_hat)
        ht = self.gate_ht(ht)

        # 4.C Gaussian Diffusion Graph Encoder
        # - Encodes concept mastery with Gaussian distribution
        # - Applies uncertainty-aware diffusion propagation over concept graph

        mt = self.get_concept_mastery(ht)  # (batch, seq-1, concept, dim_g)
        mt = self.gate_mt(mt)

        q_target = qt[:, 1:, :]  # (batch, seq-1, dim_q)
        edge_index, edge_weight = self.get_concept_map(q_target)
        c_target_idx = c_qt_idx[:, 1:, :]  # (batch, seq-1, max_c)
        mt_hat, g_target = self.diffuse_concept_mastery(mt, edge_index, edge_weight, c_target_idx)
        
        # 4.D IRT-based Prediction
        r_target, topk_r_target = self.get_concept_weight(q_target)  # (batch, seq-1, concept)
        output = self.predict(mt_hat, topk_r_target, q_target)

        # 4.E 对比学习分支（也在知识状态输出后添加门控）
        _, _, pos_ot_idx, pos_ut_idx, pos_ut_mask = self.get_index(question, concept, pos_score, pos_option, unchosen)
        _, _, neg_ot_idx, neg_ut_idx, neg_ut_mask = self.get_index(question, concept, neg_score, neg_option, unchosen)
        pos_dt_hat = self.fuse_response_state(pos_ot_idx, pos_score, pos_ut_idx, pos_ut_mask)  # (batch, seq, dim_q)
        _, pos_ht = self.memory_fused_state(qt, pos_dt_hat)  # (batch, seq-1, dim_q)
        pos_ht = self.gate_ht(pos_ht)
        neg_dt_hat = self.fuse_response_state(neg_ot_idx, neg_score, neg_ut_idx, neg_ut_mask)  # (batch, seq, dim_q)
        _, neg_ht = self.memory_fused_state(qt, neg_dt_hat)  # (batch, seq-1, dim_q)
        neg_ht = self.gate_ht(neg_ht)

        seq_count = torch.sum(seq_mask, dim=-1).unsqueeze(-1)
        pooled_score = torch.sum(ht * seq_mask.unsqueeze(-1), dim=1) / seq_count  # (batch, dim_q)
        pooled_pos_score = torch.sum(pos_ht * seq_mask.unsqueeze(-1), dim=1) / seq_count  # (batch, dim_q)
        pooled_neg_score = torch.sum(neg_ht * seq_mask.unsqueeze(-1), dim=1) / seq_count  # (batch, dim_q)
        pos_cossim = self.cossim(pooled_score.unsqueeze(1), pooled_pos_score.unsqueeze(0)) / self.temp  # (batch, batch)
        neg_cossim = self.cossim(pooled_score.unsqueeze(1), pooled_neg_score.unsqueeze(0)) / self.temp  # (batch, batch)
        neg_weights = torch.eye(neg_cossim.shape[0], device=neg_cossim.device)
        neg_cossim = neg_cossim + neg_weights
        inter_cossim = torch.cat([pos_cossim, neg_cossim], dim=1)  # (batch, batch*2)
        inter_label = torch.arange(inter_cossim.shape[0]).long().to(inter_cossim.device)

        
        # 图级对比学习嵌入构造（使用 mt: [B, T, C, dim_g] 和 r_target: [B, T, C]）
        graph_embed = torch.sum(mt * r_target.unsqueeze(-1), dim=2)  # (B, T, dim_g)
        pooled_graph = torch.sum(graph_embed * seq_mask.unsqueeze(-1), dim=1) / (seq_mask.sum(dim=1).unsqueeze(-1) + 1e-8)  # (B, dim_g)

        # 构造图对比损失（InfoNCE）
        sim_matrix = self.cossim(pooled_graph.unsqueeze(1), pooled_graph.unsqueeze(0)) / self.temp  # (B, B)
        sim_matrix_exp = torch.exp(sim_matrix)

        # 构造正负对比项（对角线为正样本）
        pos_sim = torch.diag(sim_matrix_exp)  # (B,)
        all_sim = torch.sum(sim_matrix_exp, dim=1)  # (B,)
        graph_contrastive_loss = -torch.log(pos_sim / (all_sim + 1e-8))
        graph_cl_loss = graph_contrastive_loss.mean()


        return output, r_target, g_target, inter_cossim, inter_label, graph_cl_loss, score_recon_pred

    def get_index(self, question, concept, score, option, unchosen):
        qt_idx = torch.where(question >= 0, question, self.num_q)  # (batch, seq)
        c_qt_idx = torch.where(concept >= 0, concept, self.num_c)  # (batch, seq, max_c)
        opt = torch.where(option >= 0, option, 0)  # (batch, seq)
        opt_one_hot = one_hot(opt, num_classes=self.num_o)
        ot_idx = torch.where(
            option >= 0,
            qt_idx * self.num_o + option,
            self.num_q * self.num_o
        )  # (batch, seq)
        # use all unchosen responses
        ut_mask = self.option_mask[qt_idx] - opt_one_hot
        ut_idx = torch.where(
            ut_mask > 0,
            (qt_idx * self.num_o).unsqueeze(-1) + self.option_range[qt_idx],
            self.num_q * self.num_o + 1
        )  # (batch, seq, num_o)

        return qt_idx, c_qt_idx, ot_idx, ut_idx, ut_mask

    def fuse_response_state(self, ot_idx, score, ut_idx, ut_mask):
        """
        Response-State Fusion Encoder
        """
        ot = self.response_emb(ot_idx)  # (batch, seq, dim_q)
        ut = self.response_emb(ut_idx)  # (batch, seq, num_o, dim_q)

        correct_mask = torch.eq(score, 1)
        wrong_mask = torch.eq(score, 0)

        ot_prime = ot.clone()
        ot_prime[correct_mask] = self.enc_correct(ot[correct_mask])
        ot_prime[wrong_mask] = self.enc_wrong(ot[wrong_mask])

        ut_prime = self.enc_unchosen(ut)
        ut_count = torch.sum(ut_mask, dim=-1, keepdim=True) + 1e-8
        ut_prime = torch.sum(ut_prime * ut_mask.unsqueeze(-1), dim=2) / ut_count

        # 双向融合：响应 + 负向响应构建状态原型
        fusion = ot_prime - self.lamb * ut_prime
        fusion_attn = self.attn_response(fusion, fusion, fusion, self.seq_len, maxout=False)

        return fusion_attn

    def memory_fused_state(self, qt, dt_hat):
        """
        Memory-Augmented Knowledge Retriever
        """
        qt_hat = self.attn_question(qt, qt, qt, self.seq_len, maxout=False)  # (batch, seq, dim_q)

        # 拼接当前题目和响应嵌入 → 状态输入序列
        fusion_input = qt_hat + dt_hat  # (batch, seq, dim_q)

        # 使用 GRU 获取时序知识状态表示
        ht, _ = self.state_gru(fusion_input)  # (batch, seq, dim_q)

        return qt_hat[:, :-1, :], ht[:, :-1, :]

    def get_concept_mastery(self, ht):
        ht_concept = (
            ht.unsqueeze(2)
            .expand(-1, -1, self.num_c, -1)
            .contiguous()
        )  # (batch, seq-1, concept, dim_q)

        ci = self.concept_emb.weight  # (concept, dim_c)
        ci_batch = (
            ci[None, None, :, :]
            .expand(self.batch_size, self.seq_len - 1, -1, -1)
            .contiguous()
        )  # (batch, seq-1, concept, dim_c)

        mt = self.enc_concept(torch.cat([ht_concept, ci_batch], dim=-1))  # (batch, seq-1, concept, dim_g)

        return mt

    def get_concept_map(self, q_target):
        batch_adj = (
            self.concept_map.expand(self.batch_size * (self.seq_len - 1), -1, -1)
            .contiguous()
            .to_sparse()
        )  # (batch*seq-1, concept, concept)
        batch_edge_index, edge_weight = to_edge_index(batch_adj)
        batch_index = batch_edge_index[0]
        edge_index = batch_edge_index[1:] + (batch_index * self.num_c)

        # Target-specific Edge Weight
        q_target_edge = (
            q_target.unsqueeze(2)
            .expand(-1, -1, self.num_edge, -1)
            .contiguous()
        )  # (batch, seq-1, edge, dim_q)
        cij_idx = to_edge_index(self.concept_map.to_sparse())[0]  # (2, edge)
        cij = self.concept_emb(cij_idx)  # (2, edge, dim_c)
        cij_concat = torch.cat([cij[0, :, :], cij[1, :, :]], dim=-1)  # (edge, dim_c*2)
        cij_batch = (
            cij_concat[None, None, :, :]
            .expand(self.batch_size, self.seq_len - 1, -1, -1)
            .contiguous()
        )  # (batch, seq-1, edge, dim_c*2)

        edge_weight = self.relu(self.enc_intensity(torch.cat([q_target_edge, cij_batch], dim=-1)))
        edge_weight = edge_weight.flatten()

        num_nodes = self.batch_size * (self.seq_len - 1) * self.num_c
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight,
                                                           fill_value=0.5, num_nodes=num_nodes)

        return edge_index, edge_weight    
    
    
    def diffuse_concept_mastery(self, mt, edge_index, edge_weight, c_target_idx):
        g_target = torch.sum(one_hot(c_target_idx, num_classes=self.num_c + 1).float(),
                             dim=2)[:, :, :-1]  # (batch, seq-1, concept)

        B, T, C = g_target.shape
        ht_expand = mt  # (B, T, C, dim_q)
        ci = self.concept_emb.weight
        ci_expand = ci.unsqueeze(0).unsqueeze(0).expand(B, T, C, -1).contiguous()

        ht_flat = ht_expand.reshape(-1, self.dim_q)
        ci_flat = ci_expand.reshape(-1, self.dim_c)

        # Gaussian encoding + diffusion propagation
        x = self.diff_encoder(ht_flat, ci_flat)  # (B*T*C, dim_g)
        x = x.view(B * T * C, self.dim_g)
        mt_hat = self.ggn(x=x, edge_index=edge_index, edge_weight=edge_weight)
        mt_hat = mt_hat.view(B, T, C)

        return mt_hat, g_target

    

    def get_concept_weight(self, q_target):
        ck = self.concept_emb.weight  # (concept, dim_c)
        ck_batch = (
            ck[None, None, :, :]
            .expand(self.batch_size, self.seq_len - 1, -1, -1)
            .contiguous()
        )  # (batch, seq-1, concept, dim_c)

        q_target_batch = (
            # q_target
            self.w_relevance(q_target)
            .unsqueeze(2)
            .expand(-1, -1, self.num_c, -1)
            .contiguous()
        )  # (batch, seq-1, concept, dim_c)

        r_target = torch.sum(q_target_batch * ck_batch, dim=-1)  # (batch, seq-1, concept)
        # r_target = self.sigmoid(torch.sum(q_target_batch * ck_batch, dim=-1))  # (batch, seq-1, concept)

        # Top-K Concept
        topk_values, _ = torch.topk(r_target, k=self.top_k, dim=-1, largest=True, sorted=True)  # (batch, seq-1, k)
        topk_mask = r_target >= topk_values[:, :, -1].unsqueeze(-1)
        topk_r_target = r_target.masked_fill(topk_mask == 0, -1e+32)

        topk_r_target = torch.softmax(topk_r_target, dim=-1)
        topk_r_target = topk_r_target.masked_fill(topk_mask == 0, 0)
        r_target = self.sigmoid(r_target)

        return r_target, topk_r_target

    def predict(self, mt_hat, r_target, q_target):
        ability = torch.sum(r_target * mt_hat, dim=2)  # (batch, seq-1)

        difficulty = self.mlp_diff(q_target).squeeze(-1)  # (batch, seq-1)
        # difficulty = self.sigmoid(self.mlp_diff(q_target).squeeze(-1))  # (batch, seq-1)

        output = self.sigmoid(ability - difficulty)  # (batch, seq-1)

        return output


class GraDKTLayer(nn.Module):
    def __init__(self, dim, dim_out, num_heads, dropout, kq_same=True):
        super().__init__()
        self.attn_kt = KTAttention(dim, dim_out, num_heads, kq_same)

        self.dropout = Dropout(dropout)
        self.layer_norm = LayerNorm(dim_out)

    def forward(self, query, key, value, seq_len, maxout=False):
        attn_mask = torch.tril(torch.ones(seq_len, seq_len, device=query.device), diagonal=0).bool()

        attn_output = self.attn_kt(query, key, value, attn_mask, maxout)

        output = self.layer_norm(self.dropout(attn_output))

        return output


class KTAttention(nn.Module):
    def __init__(self, dim, dim_out, num_heads, kq_same, bias=True):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.d_k = dim // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(self.dim, self.dim, bias=bias)
        if kq_same:
            self.k_linear = self.q_linear
        else:
            self.k_linear = nn.Linear(self.dim, self.dim, bias=bias)
        self.v_linear = nn.Linear(self.dim, self.dim, bias=bias)

        self.decay_rate = nn.Parameter(torch.zeros(num_heads, 1, 1))
        nn.init.xavier_uniform_(self.decay_rate)
        self.out_proj = nn.Linear(self.dim, self.dim_out, bias=bias)

        self.relu = ReLU()

    def forward(self, query, key, value, mask, maxout=False):
        self.batch_size, self.seq_len, _ = query.shape

        # perform linear operation and split into h heads
        q = self.q_linear(query).view(self.batch_size, self.seq_len, self.num_heads, self.d_k)
        k = self.k_linear(key).view(self.batch_size, self.seq_len, self.num_heads, self.d_k)
        v = self.v_linear(value).view(self.batch_size, self.seq_len, self.num_heads, self.d_k)

        q = q.transpose(1, 2)  # (batch, head, seq, d_K)
        k = k.transpose(1, 2)  # (batch, head, seq, d_K)
        v = v.transpose(1, 2)  # (batch, head, seq, d_K)

        attn_output = self.attention(q, k, v, mask, maxout)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(self.batch_size, self.seq_len, self.dim)
        )

        output = self.out_proj(attn_output)

        return output

    def attention(self, q, k, v, mask, maxout):
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.d_k)  # (batch, head, seq, seq)

        t = torch.arange(self.seq_len).float().expand(self.seq_len, -1).to(q.device)
        tau = t.transpose(0, 1).contiguous()

        # calculate context-aware attention score
        with torch.no_grad():
            # $|t - \tau|$
            temporal_dist = torch.abs(t - tau)[None, None, :, :]  # (1, 1, seq, seq)

            # $\gamma_{t, t^\prime}$
            masked_attn_score = softmax(attn_score.masked_fill(mask == 0, -1e+32), dim=-1)

            # $\sum_{t^\prime=\tau+1}^{t}{t, t^\prime}$
            tau_sum_score = torch.cumsum(masked_attn_score, dim=-1)
            t_sum_score = torch.sum(masked_attn_score, dim=-1, keepdim=True)

            dist = self.relu(temporal_dist * (t_sum_score - tau_sum_score))
            dist = dist.detach()

        total_effect = torch.exp(-self.decay_rate.abs().unsqueeze(0) * dist)
        attn_score *= total_effect

        # normalize attention score
        alpha = softmax(attn_score.masked_fill(mask == 0, -1e+32), dim=-1)
        alpha = alpha.masked_fill(mask == 0, 0)

        # maxout scale
        if maxout:
            scale = torch.clamp(1.0 / alpha.max(dim=-1, keepdim=True)[0], max=5.0)
            alpha *= scale

        output = torch.matmul(alpha, v)

        return output