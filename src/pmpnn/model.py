from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import e3nn.o3

from .constants import (
    NUM_AA,
    OA_BESSEL_EPS,
    OA_GATE_HIDDEN_DIM,
    OA_L_MAX,
    OA_MASK_FILL_VALUE,
    OA_NUM_RADIAL,
    OA_P_CUTOFF,
    OA_R_MAX,
    OA_WEIGHT_MLP_HIDDEN_DIM,
    GAT_LEAKY_RELU_SLOPE,
    PMPNN_AUGMENT_EPS,
    PMPNN_DIST_EPS,       # missing — see note at bottom of file
    PMPNN_CB_COEFF_A,     # missing — see note at bottom of file
    PMPNN_CB_COEFF_B,     # missing — see note at bottom of file
    PMPNN_CB_COEFF_C,     # missing — see note at bottom of file
    PMPNN_DROPOUT,
    PMPNN_ENC_LAYER_SCALE,
    PMPNN_FF_EXPANSION,
    PMPNN_HIDDEN_DIM,
    PMPNN_K_NEIGHBORS,
    PMPNN_MAX_RELATIVE_FEATURE,
    PMPNN_NUM_DECODER_LAYERS,
    PMPNN_NUM_ENCODER_LAYERS,
    PMPNN_NUM_POSITIONAL_EMBEDDINGS,
    PMPNN_NUM_RBF,
    PMPNN_RBF_D_MIN,
    PMPNN_RBF_D_MAX,
)


# -- Output dataclass ----------------------------------------------------------

@dataclass
class ProteinMPNNOutput:
    log_probs:       torch.Tensor
    E_idx:           torch.Tensor
    R_cut:           torch.Tensor
    Y:               torch.Tensor
    mask_attend:     torch.Tensor
    attn_weights:    List[torch.Tensor] = field(default_factory=list)
    h_E_layers:      List[torch.Tensor] = field(default_factory=list)
    spectral_layers: List[torch.Tensor] = field(default_factory=list)


# -- Utility functions ---------------------------------------------------------

def gather_edges(edges, neighbor_idx):
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, neighbors)


def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.reshape(neighbor_idx.shape[0], -1)
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    return neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])


def gather_nodes_t(nodes, neighbor_idx):
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    return torch.gather(nodes, 1, idx_flat)


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    return torch.cat([h_neighbors, h_nodes], -1)


def backbone_frame(X: torch.Tensor) -> torch.Tensor:
    """
    Per-residue orthonormal backbone frame from N / Cα / C atoms.

    X : [B, N, 4, 3]  atom order N / Cα / C / O
    Returns O : [B, N, 3, 3]  rows = (e1, e2, e3)
        e1 = normalize(Cα − N)
        e2 = normalize(e1 × (C − Cα))   normal to the N-Cα-C plane
        e3 = e1 × e2
    Invariant to global rigid-body rotations: the frame rotates equivariantly,
    so any vector expressed in it is rotation-invariant.
    """
    N_pos = X[:, :, 0, :]
    Ca    = X[:, :, 1, :]
    C_pos = X[:, :, 2, :]
    e1 = F.normalize(Ca - N_pos, dim=-1)
    e2 = F.normalize(torch.cross(e1, F.normalize(C_pos - Ca, dim=-1), dim=-1), dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-2)  # [B, N, 3, 3]


def virtual_cbeta(X: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct virtual Cβ positions from backbone N / Cα / C atoms.

    X  : [B, N, 4, 3]  atom order N / Cα / C / O
    Returns Cb : [B, N, 3]
    Coefficients (PMPNN_CB_COEFF_A/B/C) match the original ProteinMPNN formula.
    """
    b  = X[:, :, 1, :] - X[:, :, 0, :]   # Cα − N
    c  = X[:, :, 2, :] - X[:, :, 1, :]   # C  − Cα
    a  = torch.cross(b, c, dim=-1)
    return PMPNN_CB_COEFF_A * a + PMPNN_CB_COEFF_B * b + PMPNN_CB_COEFF_C * c + X[:, :, 1, :]


# -- Radial basis & cutoff -----------------------------------------------------

class BesselBasis(nn.Module):
    def __init__(self, num_basis: int, r_max: float, trainable: bool = False):
        super().__init__()
        self.r_max = r_max
        freqs = torch.arange(1, num_basis + 1, dtype=torch.float) * math.pi / r_max
        if trainable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)
        self.register_buffer("norm", torch.tensor((2.0 / r_max) ** 0.5))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        d   = r.unsqueeze(-1)
        arg = d * self.freqs
        return self.norm * torch.where(
            d.abs() < OA_BESSEL_EPS,
            self.freqs.expand_as(arg),
            torch.sin(arg) / (d + OA_BESSEL_EPS),
        )


class PolynomialCutoff(nn.Module):
    def __init__(self, r_max: float, p: int = OA_P_CUTOFF):
        super().__init__()
        self.r_max = r_max
        self.register_buffer("c1", torch.tensor((p + 1) * (p + 2) / 2.0))
        self.register_buffer("c2", torch.tensor(float(p * (p + 2))))
        self.register_buffer("c3", torch.tensor(p * (p + 1) / 2.0))
        self.p = p

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        x = (r / self.r_max).clamp(0.0, 1.0)
        return (
            1.0 - self.c1 * x.pow(self.p)
                + self.c2 * x.pow(self.p + 1)
                - self.c3 * x.pow(self.p + 2)
        ) * (r <= self.r_max).float()


# -- Orbital Attention ---------------------------------------------------------

class OAAttention(nn.Module):
    """
    Geometry-aware, rotation-invariant attention over the [B,N,K] neighbourhood.
    Scores are built from backbone-framed spherical harmonics contracted with
    radial-basis-weighted channel filters — invariant to global rigid-body rotations.
    """

    def __init__(
        self,
        feature_dim:           int,
        num_radial:            int = OA_NUM_RADIAL,
        l_max:                 int = OA_L_MAX,
        weight_mlp_hidden_dim: int = OA_WEIGHT_MLP_HIDDEN_DIM,
        gate_hidden_dim:       int = OA_GATE_HIDDEN_DIM,
    ):
        super().__init__()
        self.num_radial = num_radial
        self.n_sh       = (l_max + 1) ** 2

        self.weight_mlp = nn.Sequential(
            nn.Linear(feature_dim, weight_mlp_hidden_dim),               nn.SiLU(),
            nn.Linear(weight_mlp_hidden_dim, weight_mlp_hidden_dim // 2), nn.SiLU(),
            nn.Linear(weight_mlp_hidden_dim // 2, num_radial * self.n_sh),
        )

        self.degree_gate = nn.Sequential(
            nn.Linear(feature_dim, gate_hidden_dim), nn.SiLU(),
            nn.Linear(gate_hidden_dim, l_max + 1),
            nn.Sigmoid(),
        )
        deg_idx = [l for l in range(l_max + 1) for _ in range(2 * l + 1)]
        self.register_buffer("degree_index", torch.tensor(deg_idx, dtype=torch.long))

    def forward(
        self,
        R_cut:         torch.Tensor,   # [B, N, K, num_radial]
        Y:             torch.Tensor,   # [B, N, K, n_sh]
        edge_features: torch.Tensor,   # [B, N, K, feature_dim]
        mask_attend:   torch.Tensor,   # [B, N, K]
    ):                                 # -> (alpha [B,N,K], spectral [B,N,K,n_sh])
        B, N, K = mask_attend.shape

        W = self.weight_mlp(edge_features).view(B, N, K, self.num_radial, self.n_sh)

        g = self.degree_gate(edge_features)[..., self.degree_index]
        W = W * g.unsqueeze(-2)

        spectral = torch.einsum("bnkr,bnkrs->bnks", R_cut, W)
        scores   = (spectral * Y).sum(-1)

        scores = scores.masked_fill(mask_attend == 0, OA_MASK_FILL_VALUE)
        alpha  = F.softmax(scores, dim=-1) * mask_attend
        return alpha, spectral


# -- GAT attention -------------------------------------------------------------

class GATAttention(nn.Module):
    """Edge-driven attention without geometry. Ablation counterpart of OAAttention."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.W   = nn.Linear(feature_dim, feature_dim, bias=True)
        self.a   = nn.Linear(feature_dim, 1, bias=False)
        self.act = nn.LeakyReLU(GAT_LEAKY_RELU_SLOPE)

    def forward(
        self,
        h_E:         torch.Tensor,  # [B, N, K, feature_dim]
        mask_attend: torch.Tensor,  # [B, N, K]
    ) -> torch.Tensor:              # [B, N, K]
        scores = self.a(self.act(self.W(h_E))).squeeze(-1).float()
        scores = scores.masked_fill(mask_attend == 0, OA_MASK_FILL_VALUE)
        return (F.softmax(scores, dim=-1) * mask_attend).to(h_E.dtype)


# -- Sub-modules ---------------------------------------------------------------

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=PMPNN_MAX_RELATIVE_FEATURE):
        super().__init__()
        self.num_embeddings       = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset, mask):
        d = (
            torch.clamp(offset + self.max_relative_feature,
                        0, 2 * self.max_relative_feature) * mask
            + (1 - mask) * (2 * self.max_relative_feature + 1)
        )
        d_onehot = F.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        return self.linear(d_onehot.float())


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super().__init__()
        self.W_in  = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act   = nn.GELU()

    def forward(self, h_V):
        return self.W_out(self.act(self.W_in(h_V)))


# -- ProteinFeatures -----------------------------------------------------------

class ProteinFeatures(nn.Module):
    """
    Original ProteinFeatures augmented with local_geometry() for OAAttention.
    All original parameter names and shapes are preserved.
    """

    def __init__(
        self,
        edge_features,
        node_features,
        num_positional_embeddings: int   = PMPNN_NUM_POSITIONAL_EMBEDDINGS,
        num_rbf:                   int   = PMPNN_NUM_RBF,
        top_k:                     int   = PMPNN_K_NEIGHBORS,
        augment_eps:               float = 0.0,
        num_radial_orb: int   = OA_NUM_RADIAL,
        r_max_orb:      float = OA_R_MAX,
        p_cutoff:       int   = OA_P_CUTOFF,
        l_max:          int   = OA_L_MAX,
    ):
        super().__init__()
        self.top_k       = top_k
        self.augment_eps = augment_eps
        self.num_rbf     = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.l_max = l_max
        self.n_sh  = (l_max + 1) ** 2

        self.embeddings     = PositionalEncodings(num_positional_embeddings)
        edge_in             = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges     = nn.LayerNorm(edge_features)

        self.bessel = BesselBasis(num_radial_orb, r_max_orb, trainable=False)
        self.cutoff = PolynomialCutoff(r_max_orb, p_cutoff)

    def _dist(self, X, mask, eps=PMPNN_DIST_EPS):
        mask_2D  = mask.unsqueeze(1) * mask.unsqueeze(2)
        dX       = X.unsqueeze(1) - X.unsqueeze(2)
        D        = mask_2D * torch.sqrt((dX ** 2).sum(3) + eps)
        D_max, _ = D.max(-1, keepdim=True)
        D_adj    = D + (1.0 - mask_2D) * D_max
        D_nb, E_idx = torch.topk(
            D_adj, min(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_nb, E_idx

    def _rbf(self, D):
        D_mu    = torch.linspace(PMPNN_RBF_D_MIN, PMPNN_RBF_D_MAX, self.num_rbf, device=D.device)
        D_mu    = D_mu.view(1, 1, 1, -1)
        D_sigma = (PMPNN_RBF_D_MAX - PMPNN_RBF_D_MIN) / self.num_rbf
        return torch.exp(-((D.unsqueeze(-1) - D_mu) / D_sigma) ** 2)

    def _get_rbf(self, A, B, E_idx):
        D  = torch.sqrt(
            ((A[:, :, None, :] - B[:, None, :, :]) ** 2).sum(-1) + PMPNN_DIST_EPS)
        Dn = gather_edges(D[:, :, :, None], E_idx)[:, :, :, 0]
        return self._rbf(Dn)

    def local_geometry(self, X, E_idx):
        """
        Returns R_cut [B,N,K,R], Y [B,N,K,S], in_cutoff [B,N,K].
        Projects Cα-Cα displacement vectors into the per-residue backbone_frame.
        """
        B, N, K = E_idx.shape
        Ca = X[:, :, 1, :]
        O  = backbone_frame(X)             # [B, N, 3, 3]

        Ca_j    = gather_nodes(Ca, E_idx)
        rel_g   = Ca_j - Ca.unsqueeze(-2)
        rel_loc = torch.einsum("bnij,bnkj->bnki", O, rel_g)

        r     = rel_loc.norm(dim=-1).clamp(min=OA_BESSEL_EPS)
        d_hat = rel_loc / r.unsqueeze(-1)

        R_flat    = self.bessel(r.reshape(-1))
        u_flat    = self.cutoff(r.reshape(-1))
        R_cut     = (R_flat * u_flat.unsqueeze(-1)).reshape(B, N, K, -1)
        in_cutoff = (u_flat > 0).reshape(B, N, K)

        Y = e3nn.o3.spherical_harmonics(
            list(range(self.l_max + 1)),
            d_hat.reshape(-1, 3),
            normalize=False,
            normalization="component",
        ).reshape(B, N, K, self.n_sh)

        return R_cut, Y, in_cutoff

    def forward(self, X, mask, residue_idx, chain_labels):
        if self.augment_eps > 0 and self.training:
            X = X + self.augment_eps * torch.randn_like(X)

        Cb = virtual_cbeta(X)
        Ca = X[:, :, 1, :]
        N  = X[:, :, 0, :]
        C  = X[:, :, 2, :]
        O  = X[:, :, 3, :]

        D_nb, E_idx = self._dist(Ca, mask)

        RBF = torch.cat([
            self._rbf(D_nb),
            self._get_rbf(N,  N,  E_idx),
            self._get_rbf(C,  C,  E_idx),
            self._get_rbf(O,  O,  E_idx),
            self._get_rbf(Cb, Cb, E_idx),
            self._get_rbf(Ca, N,  E_idx),
            self._get_rbf(Ca, C,  E_idx),
            self._get_rbf(Ca, O,  E_idx),
            self._get_rbf(Ca, Cb, E_idx),
            self._get_rbf(N,  C,  E_idx),
            self._get_rbf(N,  O,  E_idx),
            self._get_rbf(N,  Cb, E_idx),
            self._get_rbf(Cb, C,  E_idx),
            self._get_rbf(Cb, O,  E_idx),
            self._get_rbf(O,  C,  E_idx),
            self._get_rbf(N,  Ca, E_idx),
            self._get_rbf(C,  Ca, E_idx),
            self._get_rbf(O,  Ca, E_idx),
            self._get_rbf(Cb, Ca, E_idx),
            self._get_rbf(C,  N,  E_idx),
            self._get_rbf(O,  N,  E_idx),
            self._get_rbf(Cb, N,  E_idx),
            self._get_rbf(C,  Cb, E_idx),
            self._get_rbf(O,  Cb, E_idx),
            self._get_rbf(C,  O,  E_idx),
        ], dim=-1)

        offset   = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset   = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]
        d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).long()
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_pos    = self.embeddings(offset.long(), E_chains)

        E = self.edge_embedding(torch.cat([E_pos, RBF], dim=-1))
        E = self.norm_edges(E)
        return E, E_idx


# -- EncLayer ------------------------------------------------------------------

class EncLayer(nn.Module):
    """
    Encoder layer with three aggregation modes:
      - OAAttention  (use_orbital_attn=True)
      - GATAttention (use_gat=True)
      - plain sum/scale (vanilla baseline)
    """

    def __init__(
        self,
        num_hidden,
        num_in,
        dropout    = PMPNN_DROPOUT,
        num_heads  = None,
        scale      = PMPNN_ENC_LAYER_SCALE,
        num_radial:            int  = OA_NUM_RADIAL,
        l_max:                 int  = OA_L_MAX,
        use_orbital_attn:      bool = True,
        weight_mlp_hidden_dim: int  = OA_WEIGHT_MLP_HIDDEN_DIM,
        gate_hidden_dim:       int  = OA_GATE_HIDDEN_DIM,
        use_gat:               bool = False,
    ):
        super().__init__()
        self.num_hidden       = num_hidden
        self.scale            = scale
        self.use_orbital_attn = use_orbital_attn
        self.use_gat          = use_gat

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1    = nn.LayerNorm(num_hidden)
        self.norm2    = nn.LayerNorm(num_hidden)
        self.norm3    = nn.LayerNorm(num_hidden)
        self.W1  = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2  = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3  = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act  = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * PMPNN_FF_EXPANSION)

        if use_orbital_attn:
            self.orbital_attn = OAAttention(
                feature_dim           = num_hidden,
                num_radial            = num_radial,
                l_max                 = l_max,
                weight_mlp_hidden_dim = weight_mlp_hidden_dim,
                gate_hidden_dim       = gate_hidden_dim,
            )

        if use_gat:
            self.gat_attn = GATAttention(feature_dim=num_hidden)

    def forward(self, h_V, h_E, E_idx, R_cut, Y, mask_V=None, mask_attend=None):
        h_EV       = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV       = torch.cat([h_V_expand, h_EV], -1)
        h_message  = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))

        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message

        if self.use_orbital_attn:
            ma              = (mask_attend if mask_attend is not None
                               else h_message.new_ones(h_message.shape[:3]))
            alpha, spectral = self.orbital_attn(R_cut, Y, h_E, ma)
            dh              = (alpha.unsqueeze(-1) * h_message).sum(-2)
        elif self.use_gat:
            ma       = (mask_attend if mask_attend is not None
                        else h_message.new_ones(h_message.shape[:3]))
            alpha    = self.gat_attn(h_E, ma)
            spectral = None
            dh       = (alpha.unsqueeze(-1) * h_message).sum(-2)
        else:
            alpha    = None
            spectral = None
            dh       = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))
        dh  = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            h_V = mask_V.unsqueeze(-1) * h_V

        h_EV       = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV       = torch.cat([h_V_expand, h_EV], -1)
        h_message  = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E        = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E, alpha, spectral


# -- DecLayer (unchanged) ------------------------------------------------------

class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=PMPNN_DROPOUT, num_heads=None, scale=PMPNN_ENC_LAYER_SCALE):
        super().__init__()
        self.num_hidden = num_hidden
        self.scale      = scale
        self.dropout1   = nn.Dropout(dropout)
        self.dropout2   = nn.Dropout(dropout)
        self.norm1      = nn.LayerNorm(num_hidden)
        self.norm2      = nn.LayerNorm(num_hidden)
        self.W1    = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2    = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3    = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act   = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * PMPNN_FF_EXPANSION)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV       = torch.cat([h_V_expand, h_E], -1)
        h_message  = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh  = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))
        dh  = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            h_V = mask_V.unsqueeze(-1) * h_V
        return h_V


# -- ProteinMPNN ---------------------------------------------------------------

class ProteinMPNN(nn.Module):
    """
    ProteinMPNN augmented with Orbital Attention (OAAttention).

    Modes (set via use_orbital_attn / use_vanilla / use_gat):
      vanilla — original sum/scale aggregation
      oa      — OAAttention with degree gating (main model)
      gat     — GATAttention, geometry-free ablation

    Weight compatibility: original ProteinMPNN checkpoint keys are preserved.
    New keys live under features.bessel.*, features.cutoff.*,
    encoder_layers.*.orbital_attn.*, encoder_layers.*.gat_attn.*.
    """

    def __init__(
        self,
        num_letters:          int   = NUM_AA,
        node_features:        int   = PMPNN_HIDDEN_DIM,
        edge_features:        int   = PMPNN_HIDDEN_DIM,
        hidden_dim:           int   = PMPNN_HIDDEN_DIM,
        num_encoder_layers:   int   = PMPNN_NUM_ENCODER_LAYERS,
        num_decoder_layers:   int   = PMPNN_NUM_DECODER_LAYERS,
        vocab:                int   = NUM_AA,
        k_neighbors:          int   = PMPNN_K_NEIGHBORS,
        augment_eps:          float = PMPNN_AUGMENT_EPS,
        dropout:              float = PMPNN_DROPOUT,
        num_radial_orb:        int   = OA_NUM_RADIAL,
        r_max_orb:             float = OA_R_MAX,
        p_cutoff:              int   = OA_P_CUTOFF,
        l_max:                 int   = OA_L_MAX,
        use_orbital_attn:      bool  = True,
        weight_mlp_hidden_dim: int   = OA_WEIGHT_MLP_HIDDEN_DIM,
        gate_hidden_dim:       int   = OA_GATE_HIDDEN_DIM,
        use_vanilla:           bool  = False,
        use_gat:               bool  = False,
    ):
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim    = hidden_dim
        self.use_vanilla   = use_vanilla
        self.use_gat       = use_gat

        self.features = ProteinFeatures(
            node_features, edge_features,
            top_k          = k_neighbors,
            augment_eps    = augment_eps,
            num_radial_orb = num_radial_orb,
            r_max_orb      = r_max_orb,
            p_cutoff       = p_cutoff,
            l_max          = l_max,
        )

        self.W_e   = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s   = nn.Embedding(vocab, hidden_dim)
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        self.encoder_layers = nn.ModuleList([
            EncLayer(
                hidden_dim, hidden_dim * 2,
                dropout               = dropout,
                num_radial            = num_radial_orb,
                l_max                 = l_max,
                use_orbital_attn      = use_orbital_attn and not use_vanilla and not use_gat,
                weight_mlp_hidden_dim = weight_mlp_hidden_dim,
                gate_hidden_dim       = gate_hidden_dim,
                use_gat               = use_gat,
            )
            for _ in range(num_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for layer in self.encoder_layers:
            if hasattr(layer, 'orbital_attn'):
                nn.init.orthogonal_(layer.orbital_attn.weight_mlp[0].weight)
                nn.init.orthogonal_(layer.orbital_attn.weight_mlp[2].weight)

    def _encode(self, X, mask, residue_idx, chain_encoding_all):
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_E = self.W_e(E)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=X.device)

        if self.use_vanilla or self.use_gat:
            mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
            mask_attend = mask.unsqueeze(-1) * mask_attend
            R_cut = Y = None
        else:
            R_cut, Y, in_cutoff = self.features.local_geometry(X, E_idx)

            K_eff     = int(in_cutoff.sum(-1).max())
            E_idx     = E_idx[:, :, :K_eff]
            h_E       = h_E[:, :, :K_eff]
            R_cut     = R_cut[:, :, :K_eff]
            Y         = Y[:, :, :K_eff]
            in_cutoff = in_cutoff[:, :, :K_eff]

            mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
            mask_attend = mask.unsqueeze(-1) * mask_attend * in_cutoff

        for layer in self.encoder_layers:
            h_V, h_E, *_ = layer(h_V, h_E, E_idx, R_cut, Y,
                                 mask_V=mask, mask_attend=mask_attend)
        return h_V, h_E, E_idx

    def _encode_with_attn(self, X, mask, residue_idx, chain_encoding_all):
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_E = self.W_e(E)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=X.device)

        if self.use_vanilla or self.use_gat:
            mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
            mask_attend = mask.unsqueeze(-1) * mask_attend
            R_cut = Y = None
        else:
            R_cut, Y, in_cutoff = self.features.local_geometry(X, E_idx)

            K_eff     = int(in_cutoff.sum(-1).max())
            E_idx     = E_idx[:, :, :K_eff]
            h_E       = h_E[:, :, :K_eff]
            R_cut     = R_cut[:, :, :K_eff]
            Y         = Y[:, :, :K_eff]
            in_cutoff = in_cutoff[:, :, :K_eff]

            mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
            mask_attend = mask.unsqueeze(-1) * mask_attend * in_cutoff

        attn_weights:    List[torch.Tensor] = []
        h_E_layers:      List[torch.Tensor] = []
        spectral_layers: List[torch.Tensor] = []
        for layer in self.encoder_layers:
            h_E_layers.append(h_E)
            h_V, h_E, alpha, spectral = layer(h_V, h_E, E_idx, R_cut, Y,
                                               mask_V=mask, mask_attend=mask_attend)
            attn_weights.append(alpha)
            spectral_layers.append(spectral)

        return h_V, h_E, E_idx, R_cut, Y, mask_attend, attn_weights, h_E_layers, spectral_layers

    def forward(
        self,
        X,
        S,
        mask,
        chain_M,
        residue_idx,
        chain_encoding_all,
        randn,
        use_input_decoding_order: bool = False,
        decoding_order            = None,
    ) -> ProteinMPNNOutput:
        device = X.device

        (h_V, h_E, E_idx,
         R_cut, Y, enc_mask_attend,
         attn_weights, h_E_layers, spectral_layers) = self._encode_with_attn(
            X, mask, residue_idx, chain_encoding_all)

        h_S           = self.W_s(S)
        h_ES          = cat_neighbors_nodes(h_S, h_E, E_idx)
        h_EX_encoder  = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M * mask
        if not use_input_decoding_order:
            decoding_order = torch.argsort(
                (chain_M + 0.0001) * torch.abs(randn))

        mask_size = E_idx.shape[1]
        P_rev     = F.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_bw = torch.einsum(
            "ij,biq,bjp->bqp",
            1 - torch.triu(torch.ones(mask_size, mask_size, device=device)),
            P_rev, P_rev,
        )
        mask_attend   = torch.gather(order_mask_bw, 2, E_idx).unsqueeze(-1)
        mask_1D       = mask.view(mask.size(0), mask.size(1), 1, 1)
        mask_bw       = mask_1D * mask_attend
        mask_fw       = mask_1D * (1.0 - mask_attend)
        h_EXV_enc_fw  = mask_fw * h_EXV_encoder

        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_enc_fw
            h_V = layer(h_V, h_ESV, mask)

        return ProteinMPNNOutput(
            log_probs       = F.log_softmax(self.W_out(h_V), dim=-1),
            E_idx           = E_idx,
            R_cut           = R_cut,
            Y               = Y,
            mask_attend     = enc_mask_attend,
            attn_weights    = attn_weights,
            h_E_layers      = h_E_layers,
            spectral_layers = spectral_layers,
        )

    def sample(
        self,
        X, randn, S_true, chain_mask, chain_encoding_all, residue_idx,
        mask=None, temperature=1.0,
        omit_AAs_np=None, bias_AAs_np=None,
        chain_M_pos=None, omit_AA_mask=None,
        pssm_coef=None, pssm_bias=None, pssm_multi=None,
        pssm_log_odds_flag=None, pssm_log_odds_mask=None,
        pssm_bias_flag=None, bias_by_res=None,
    ):
        device = X.device

        h_V, h_E, E_idx = self._encode(X, mask, residue_idx, chain_encoding_all)

        chain_mask   = chain_mask * chain_M_pos * mask
        decoding_order = torch.argsort(
            (chain_mask + 0.0001) * torch.abs(randn))
        mask_size    = E_idx.shape[1]
        P_rev        = F.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_bw = torch.einsum(
            "ij,biq,bjp->bqp",
            1 - torch.triu(torch.ones(mask_size, mask_size, device=device)),
            P_rev, P_rev,
        )
        mask_attend  = torch.gather(order_mask_bw, 2, E_idx).unsqueeze(-1)
        mask_1D      = mask.view(mask.size(0), mask.size(1), 1, 1)
        mask_bw      = mask_1D * mask_attend
        mask_fw      = mask_1D * (1.0 - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        log_probs  = torch.zeros((N_batch, N_nodes, NUM_AA), device=device)
        all_probs  = torch.zeros((N_batch, N_nodes, NUM_AA), device=device)
        h_S        = torch.zeros_like(h_V)
        S          = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack  = [h_V] + [
            torch.zeros_like(h_V) for _ in range(len(self.decoder_layers))
        ]
        constant      = torch.tensor(omit_AAs_np, device=device)
        constant_bias = torch.tensor(bias_AAs_np,  device=device)
        omit_AA_mask_flag = omit_AA_mask is not None

        h_EX_encoder  = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_enc_fw  = mask_fw * h_EXV_encoder

        for t_ in range(N_nodes):
            t = decoding_order[:, t_]
            chain_mask_gathered = torch.gather(chain_mask, 1, t[:, None])
            mask_gathered       = torch.gather(mask, 1, t[:, None])
            bias_by_res_gathered = torch.gather(
                bias_by_res, 1,
                t[:, None, None].expand(-1, 1, NUM_AA))[:, 0, :]

            if (mask_gathered == 0).all():
                S_t = torch.gather(S_true, 1, t[:, None])
            else:
                E_idx_t = torch.gather(
                    E_idx, 1, t[:, None, None].expand(-1, 1, E_idx.shape[-1]))
                h_E_t = torch.gather(
                    h_E, 1,
                    t[:, None, None, None].expand(-1, 1, h_E.shape[-2], h_E.shape[-1]))
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                h_EXV_enc_t = torch.gather(
                    h_EXV_enc_fw, 1,
                    t[:, None, None, None].expand(
                        -1, 1, h_EXV_enc_fw.shape[-2], h_EXV_enc_fw.shape[-1]))
                mask_t = torch.gather(mask, 1, t[:, None])

                for l, layer in enumerate(self.decoder_layers):
                    h_ESV_dec_t = cat_neighbors_nodes(
                        h_V_stack[l], h_ES_t, E_idx_t)
                    h_V_t = torch.gather(
                        h_V_stack[l], 1,
                        t[:, None, None].expand(-1, 1, h_V_stack[l].shape[-1]))
                    h_ESV_t = (
                        torch.gather(
                            mask_bw, 1,
                            t[:, None, None, None].expand(
                                -1, 1, mask_bw.shape[-2], mask_bw.shape[-1]))
                        * h_ESV_dec_t + h_EXV_enc_t
                    )
                    h_V_stack[l + 1].scatter_(
                        1,
                        t[:, None, None].expand(-1, 1, h_V.shape[-1]),
                        layer(h_V_t, h_ESV_t, mask_V=mask_t),
                    )

                h_V_t   = torch.gather(
                    h_V_stack[-1], 1,
                    t[:, None, None].expand(-1, 1, h_V_stack[-1].shape[-1]))[:, 0]
                logits  = self.W_out(h_V_t) / temperature
                probs   = F.softmax(
                    logits
                    - constant[None, :] * 1e8
                    + constant_bias[None, :] / temperature
                    + bias_by_res_gathered / temperature,
                    dim=-1,
                )
                if pssm_bias_flag:
                    pssm_coef_g = torch.gather(pssm_coef, 1, t[:, None])[:, 0]
                    pssm_bias_g = torch.gather(
                        pssm_bias, 1,
                        t[:, None, None].expand(-1, 1, pssm_bias.shape[-1]))[:, 0]
                    probs = ((1 - pssm_multi * pssm_coef_g[:, None]) * probs
                             + pssm_multi * pssm_coef_g[:, None] * pssm_bias_g)
                if pssm_log_odds_flag:
                    lo_mask = torch.gather(
                        pssm_log_odds_mask, 1,
                        t[:, None, None].expand(
                            -1, 1, pssm_log_odds_mask.shape[-1]))[:, 0]
                    probs = (probs * lo_mask + probs * 0.001)
                    probs = probs / probs.sum(-1, keepdim=True)
                if omit_AA_mask_flag:
                    omit_g = torch.gather(
                        omit_AA_mask, 1,
                        t[:, None, None].expand(-1, 1, omit_AA_mask.shape[-1]))[:, 0]
                    probs  = probs * (1.0 - omit_g)
                    probs  = probs / probs.sum(-1, keepdim=True)
                S_t = torch.multinomial(probs, 1)
                all_probs.scatter_(
                    1, t[:, None, None].expand(-1, 1, NUM_AA),
                    (chain_mask_gathered[:, :, None] * probs[:, None, :]).float())

            S_true_g = torch.gather(S_true, 1, t[:, None])
            S_t      = (S_t * chain_mask_gathered
                        + S_true_g * (1.0 - chain_mask_gathered)).long()
            h_S.scatter_(
                1, t[:, None, None].expand(-1, 1, self.W_s.embedding_dim),
                self.W_s(S_t))
            S.scatter_(1, t[:, None], S_t)

        return {"S": S, "probs": all_probs, "decoding_order": decoding_order}

    def tied_sample(
        self,
        X, randn, S_true, chain_mask, chain_encoding_all, residue_idx,
        mask=None, temperature=1.0,
        omit_AAs_np=None, bias_AAs_np=None,
        chain_M_pos=None, omit_AA_mask=None,
        pssm_coef=None, pssm_bias=None, pssm_multi=None,
        pssm_log_odds_flag=None, pssm_log_odds_mask=None,
        pssm_bias_flag=None, tied_pos=None, tied_beta=None, bias_by_res=None,
    ):
        device = X.device

        h_V, h_E, E_idx = self._encode(X, mask, residue_idx, chain_encoding_all)

        chain_mask   = chain_mask * chain_M_pos * mask
        decoding_order = torch.argsort(
            (chain_mask + 0.0001) * torch.abs(randn))

        new_decoding_order = []
        for t_dec in decoding_order[0].cpu().tolist():
            if t_dec not in list(itertools.chain(*new_decoding_order)):
                list_a = [item for item in tied_pos if t_dec in item]
                new_decoding_order.append(list_a[0] if list_a else [t_dec])
        decoding_order = torch.tensor(
            list(itertools.chain(*new_decoding_order)), device=device
        )[None].expand(X.shape[0], -1)

        mask_size    = E_idx.shape[1]
        P_rev        = F.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_bw = torch.einsum(
            "ij,biq,bjp->bqp",
            1 - torch.triu(torch.ones(mask_size, mask_size, device=device)),
            P_rev, P_rev,
        )
        mask_attend  = torch.gather(order_mask_bw, 2, E_idx).unsqueeze(-1)
        mask_1D      = mask.view(mask.size(0), mask.size(1), 1, 1)
        mask_bw      = mask_1D * mask_attend
        mask_fw      = mask_1D * (1.0 - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        all_probs  = torch.zeros((N_batch, N_nodes, NUM_AA), device=device)
        h_S        = torch.zeros_like(h_V)
        S          = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack  = [h_V] + [
            torch.zeros_like(h_V) for _ in range(len(self.decoder_layers))
        ]
        constant      = torch.tensor(omit_AAs_np, device=device)
        constant_bias = torch.tensor(bias_AAs_np,  device=device)
        omit_AA_mask_flag = omit_AA_mask is not None

        h_EX_encoder  = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_enc_fw  = mask_fw * h_EXV_encoder

        for t_list in new_decoding_order:
            logits    = 0.0
            done_flag = False
            for t in t_list:
                if (mask[:, t] == 0).all():
                    S_t = S_true[:, t]
                    for tt in t_list:
                        h_S[:, tt, :] = self.W_s(S_t)
                        S[:, tt]      = S_t
                    done_flag = True
                    break
                E_idx_t = E_idx[:, t:t+1, :]
                h_E_t   = h_E[:, t:t+1, :, :]
                h_ES_t  = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                h_EXV_enc_t = h_EXV_enc_fw[:, t:t+1, :, :]
                mask_t  = mask[:, t:t+1]

                for l, layer in enumerate(self.decoder_layers):
                    h_ESV_dec_t = cat_neighbors_nodes(
                        h_V_stack[l], h_ES_t, E_idx_t)
                    h_V_t   = h_V_stack[l][:, t:t+1, :]
                    h_ESV_t = mask_bw[:, t:t+1, :, :] * h_ESV_dec_t + h_EXV_enc_t
                    h_out   = layer(h_V_t, h_ESV_t, mask_V=mask_t)
                    h_V_stack[l + 1][:, t, :] = h_out[:, 0, :]

                h_V_t  = h_V_stack[-1][:, t, :]
                logits = logits + tied_beta[t] * (
                    self.W_out(h_V_t) / temperature
                    - constant[None, :] * 1e8
                    + constant_bias[None, :] / temperature
                    + bias_by_res[:, t, :] / temperature
                )

            if not done_flag:
                probs = F.softmax(logits, dim=-1)
                if pssm_bias_flag:
                    pssm_coef_g = pssm_coef[:, t_list[0]]
                    pssm_bias_g = pssm_bias[:, t_list[0], :]
                    probs = ((1 - pssm_multi * pssm_coef_g[:, None]) * probs
                             + pssm_multi * pssm_coef_g[:, None] * pssm_bias_g)
                if pssm_log_odds_flag:
                    lo_mask = pssm_log_odds_mask[:, t_list[0], :]
                    probs   = probs * lo_mask + probs * 0.001
                    probs   = probs / probs.sum(-1, keepdim=True)
                if omit_AA_mask_flag:
                    omit_g = omit_AA_mask[:, t_list[0], :]
                    probs  = probs * (1.0 - omit_g)
                    probs  = probs / probs.sum(-1, keepdim=True)
                S_t = torch.multinomial(probs, 1)[:, 0]
                for t in t_list:
                    h_S[:, t, :] = self.W_s(S_t)
                    S[:, t]      = S_t
                    all_probs[:, t, :] = probs.float()

        return {"S": S, "probs": all_probs, "decoding_order": decoding_order}

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        map_location = "cpu",
        **model_kwargs,
    ) -> "ProteinMPNN":
        ckpt  = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)

        model = cls(**model_kwargs)
        missing, unexpected = model.load_state_dict(state, strict=False)

        _new_tags = ("orbital_attn", "bessel", "cutoff", "gat_attn")
        new_keys      = {k for k in missing if any(t in k for t in _new_tags)}
        truly_missing = set(missing) - new_keys

        if truly_missing:
            print(f"[ProteinMPNN] WARNING — missing original keys:\n"
                  f"  {truly_missing}")
        if unexpected:
            print(f"[ProteinMPNN] WARNING — unexpected checkpoint keys:\n"
                  f"  {unexpected}")

        # Sanity-check r_max when OA is active: reconstruct it from the stored
        # Bessel frequencies (freqs[0] = π / r_max) and compare against the
        # value the caller passed in.  A mismatch means the model was trained
        # with a different cutoff, which silently corrupts the geometry.
        if not model.use_vanilla and not model.use_gat:
            bessel_key = "features.bessel.freqs"
            if bessel_key in state:
                r_max_ckpt = math.pi / state[bessel_key][0].item()
                r_max_used = model.features.bessel.r_max
                if abs(r_max_ckpt - r_max_used) > 1e-3:
                    print(
                        f"[ProteinMPNN] WARNING — r_max mismatch: "
                        f"checkpoint was trained with r_max={r_max_ckpt:.4f} Å "
                        f"but r_max_orb={r_max_used:.4f} Å was passed. "
                        f"Pass r_max_orb={r_max_ckpt:.4f} to from_pretrained()."
                    )
                else:
                    print(f"[ProteinMPNN] r_max check passed ({r_max_ckpt:.4f} Å).")

        n_loaded = len(state) - len(unexpected)
        print(f"[ProteinMPNN] Loaded {n_loaded} original parameters. "
              f"{len(new_keys)} new OA parameters randomly initialised.")
        return model

    def freeze_backbone(self):
        _new_tags = ("orbital_attn", "bessel", "cutoff", "gat_attn")
        for name, param in self.named_parameters():
            param.requires_grad = any(t in name for t in _new_tags)
        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[ProteinMPNN] Frozen {n_frozen:,} | Trainable {n_train:,} params.")

    def freeze_encoder(self):
        _new_tags = ("orbital_attn", "bessel", "cutoff", "gat_attn")
        for name, param in self.encoder_layers.named_parameters():
            param.requires_grad = any(t in name for t in _new_tags)
        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[ProteinMPNN] Frozen {n_frozen:,} | Trainable {n_train:,} params.")

    def freeze_decoder(self):
        for param in self.decoder_layers.parameters():
            param.requires_grad = False
        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[ProteinMPNN] Frozen {n_frozen:,} | Trainable {n_train:,} params.")

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
        total = sum(p.numel() for p in self.parameters())
        print(f"[ProteinMPNN] All {total:,} parameters unfrozen.")
