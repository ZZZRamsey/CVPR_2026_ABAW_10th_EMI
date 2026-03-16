"""
model_ablation.py – Controlled ablation model for ABAW-EMI.

Cross-Attention baseline with individually togglable components.
Default (all flags=False) reproduces abaw_ca/model.py behaviour.

Flags
-----
use_projection       : LN+Linear+GELU project each modality to hidden_dim
                       before cross-attention.
use_vision_temporal  : depthwise-Conv1d + TransformerEncoder temporal
                       adapter on vision.  [requires use_projection=True]
use_fusion_self_attn : Self-attention over modality tokens instead of
                       concat+MLP fusion.  [requires use_projection=True]
use_sigmoid          : sigmoid on regression output.
use_gate_intensity   : dual gate+intensity head (returns tuple).
freeze_encoders      : freeze wav2vec2 + text encoder weights.
"""
import os, math, pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from abaw.audeer import EmotionModel


class CrossAttentionLayer(nn.Module):
    """Text global-token as query, temporal modality features as key/value."""
    def __init__(self, query_dim, kv_dim, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.scale     = math.sqrt(self.head_dim)
        self.q_proj    = nn.Linear(query_dim, hidden_dim)
        self.k_proj    = nn.Linear(kv_dim, hidden_dim)
        self.v_proj    = nn.Linear(kv_dim, hidden_dim)
        self.out_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.ln_q      = nn.LayerNorm(hidden_dim)
        self.ln_kv     = nn.LayerNorm(hidden_dim)
        self.drop      = nn.Dropout(dropout)

    def forward(self, query, key_value, key_padding_mask=None):
        """
        query:           [B, D_q]
        key_value:       [B, T, D_kv]
        key_padding_mask:[B, T]  True = padded (ignore)
        Returns:         [B, hidden_dim]
        """
        B, T, _  = key_value.shape
        q = self.ln_q(self.q_proj(query)).unsqueeze(1)  # [B,1,H]
        k = self.ln_kv(self.k_proj(key_value))          # [B,T,H]
        v = self.v_proj(key_value)                       # [B,T,H]

        H, nh, hd = q.size(-1), self.num_heads, self.head_dim
        q = q.view(B,  1, nh, hd).transpose(1, 2)
        k = k.view(B,  T, nh, hd).transpose(1, 2)
        v = v.view(B,  T, nh, hd).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B,nh,1,T]
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attn = self.drop(F.softmax(attn, dim=-1))
        out  = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, H)
        return self.out_proj(out)


class ModelAblation(nn.Module):
    """
    Multimodal cross-attention model for ABAW-EMI with individually toggleable
    components.  With all flags=False the architecture is identical to
    abaw_ca/model.py (the historical ~0.529 baseline).
    """
    def __init__(self, model_name, task=None, modality_dropout_p=0.0,
                 vision_dir=None,
                 use_projection=False,
                 use_vision_temporal=False,
                 use_fusion_self_attn=False,
                 use_sigmoid=False,
                 use_gate_intensity=False,
                 freeze_encoders=False,
                 hidden_dim=512):
        super().__init__()
        assert not (use_vision_temporal  and not use_projection), \
            "use_vision_temporal requires use_projection=True"
        assert not (use_fusion_self_attn and not use_projection), \
            "use_fusion_self_attn requires use_projection=True"

        self.task                = task
        self.modality_dropout_p  = modality_dropout_p
        self.vision_dir          = vision_dir
        self.use_projection      = use_projection
        self.use_vision_temporal = use_vision_temporal
        self.use_fusion_self_attn= use_fusion_self_attn
        self.use_sigmoid         = use_sigmoid
        self.use_gate_intensity  = use_gate_intensity
        self.hidden_dim          = hidden_dim

        # ── Pretrained encoders ─────────────────────────────────────────
        self.audio_model = EmotionModel.from_pretrained(model_name[1])
        self.text_model  = AutoModel.from_pretrained(model_name[2],
                                                     trust_remote_code=True)
        audio_feat_dim = 1027
        text_feat_dim  = 768

        # Auto-detect vision feature dim
        if vision_dir is not None:
            _vd = vision_dir
        else:
            _vd = ("data/googlevit" if os.path.exists("data/googlevit")
                   and os.listdir("data/googlevit") else "data/vit")
        _fp = os.path.join(_vd, sorted(os.listdir(_vd))[0])
        with open(_fp, "rb") as f:
            _s = pickle.load(f)
        vis_input_dim = (_s.shape[-1] if hasattr(_s, "shape")
                         else torch.tensor(_s).shape[-1])
        print(f"Vision dir: {_vd}, dim: {vis_input_dim}")

        # ── Learnable missing tokens ────────────────────────────────────
        self.miss_audio  = nn.Parameter(torch.randn(1, 1, audio_feat_dim) * 0.02)
        self.miss_vision = nn.Parameter(torch.randn(1, 1, vis_input_dim)  * 0.02)
        self.miss_text   = nn.Parameter(torch.randn(1, text_feat_dim)     * 0.02)

        # ── Optional projection ─────────────────────────────────────────
        if use_projection:
            self.audio_proj  = nn.Sequential(
                nn.LayerNorm(audio_feat_dim),
                nn.Linear(audio_feat_dim, hidden_dim), nn.GELU())
            self.vision_proj = nn.Sequential(
                nn.LayerNorm(vis_input_dim),
                nn.Linear(vis_input_dim, hidden_dim),  nn.GELU())
            self.text_proj   = nn.Sequential(
                nn.LayerNorm(text_feat_dim),
                nn.Linear(text_feat_dim, hidden_dim),  nn.GELU())
            ca_q = hidden_dim; ca_ka = hidden_dim; ca_kv = hidden_dim
        else:
            ca_q  = text_feat_dim    # 768
            ca_ka = audio_feat_dim   # 1027
            ca_kv = vis_input_dim    # 768

        # ── Optional vision temporal adapter ───────────────────────────
        if use_vision_temporal:
            self.vis_conv = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, 1), nn.GELU())
            self.vis_enc  = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4,
                    dim_feedforward=hidden_dim*2, dropout=0.1, batch_first=True),
                num_layers=2)

        # ── Cross-attention layers ──────────────────────────────────────
        if "audio" in task:
            self.ca_audio  = CrossAttentionLayer(ca_q, ca_ka, hidden_dim)
        if "vit" in task:
            self.ca_vision = CrossAttentionLayer(ca_q, ca_kv, hidden_dim)

        # ── Fusion ─────────────────────────────────────────────────────
        if use_fusion_self_attn:
            self.fusion_sa = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4,
                    dim_feedforward=hidden_dim*2, dropout=0.1, batch_first=True),
                num_layers=2)
            fusion_out = hidden_dim
        else:
            text_d     = hidden_dim if use_projection else text_feat_dim
            fusion_out = text_d
            if "audio" in task: fusion_out += hidden_dim
            if "vit"   in task: fusion_out += hidden_dim

        # ── Prediction head(s) ─────────────────────────────────────────
        if use_gate_intensity:
            self.gate_head      = nn.Sequential(
                nn.Linear(fusion_out, hidden_dim), nn.GELU(),
                nn.Dropout(0.1), nn.Linear(hidden_dim, 6))
            self.intensity_head = nn.Sequential(
                nn.Linear(fusion_out, hidden_dim), nn.GELU(),
                nn.Dropout(0.1), nn.Linear(hidden_dim, 6))
        else:
            self.pred_head = nn.Sequential(
                nn.Linear(fusion_out, hidden_dim), nn.GELU(),
                nn.Dropout(0.1), nn.Linear(hidden_dim, 6))

        # ── Freeze encoders ─────────────────────────────────────────────
        if freeze_encoders:
            for p in self.audio_model.parameters(): p.requires_grad = False
            for p in self.text_model.parameters():  p.requires_grad = False
            n = sum(p.numel() for p in self.parameters() if not p.requires_grad)
            print(f"Frozen {n:,} encoder parameters")

    # ────────────────────────────────────────────────────────────────────
    def forward(self, audio, vision, text, length,
                vision_missing=None, text_missing=None):
        # ── Encode audio ────────────────────────────────────────────────
        raw_len = audio["attention_mask"].sum(dim=1)
        ao      = self.audio_model(audio["input_values"])
        a       = torch.cat((ao[0], ao[1]), dim=2)       # [B, T_a, 1027]
        T_a     = a.size(1)
        ds      = (12 * 16000) / T_a
        eff_len = torch.floor(raw_len.float() / ds).long().clamp(min=1)
        a_mask  = (torch.arange(T_a, device=a.device).unsqueeze(0)
                   >= eff_len.unsqueeze(1))               # True = padded

        # ── Encode text ─────────────────────────────────────────────────
        t = self.text_model(**text).last_hidden_state[:, 0, :]   # [B,768]

        # ── Explicit missing flags ──────────────────────────────────────
        if text_missing is not None:
            m = text_missing.bool()
            if m.any():
                t = t.clone()
                t[m] = self.miss_text.expand(m.sum(), -1)
        if vision_missing is not None:
            m = vision_missing.bool()
            if m.any():
                vision = vision.clone()
                vision[m] = (self.miss_vision.expand(-1, vision.size(1), -1)
                                              .expand(m.sum(), -1, -1))

        # ── Modality dropout (training only) ────────────────────────────
        if self.training and self.modality_dropout_p > 0:
            B = t.size(0)
            if "audio" in self.task:
                d = torch.rand(B, device=a.device) < self.modality_dropout_p
                if d.any():
                    a = a.clone()
                    a[d] = (self.miss_audio.expand(-1, T_a, -1)
                                           .expand(d.sum(), -1, -1))
                    a_mask[d] = False
            if "vit" in self.task:
                d = torch.rand(B, device=vision.device) < self.modality_dropout_p
                if d.any():
                    vision = vision.clone()
                    vision[d] = (self.miss_vision.expand(-1, vision.size(1), -1)
                                                  .expand(d.sum(), -1, -1))
            d = torch.rand(B, device=t.device) < self.modality_dropout_p
            if d.any():
                t = t.clone()
                t[d] = self.miss_text.expand(d.sum(), -1)

        # ── Optional projection ─────────────────────────────────────────
        if self.use_projection:
            a      = self.audio_proj(a)        # [B,T_a,512]
            t      = self.text_proj(t)         # [B,512]
            vision = self.vision_proj(vision)  # [B,T_v,512]

        # ── Optional vision temporal adapter ───────────────────────────
        if self.use_vision_temporal:
            vision = self.vis_conv(vision.transpose(1, 2)).transpose(1, 2)
            T_v    = vision.size(1)
            vm_t   = (torch.arange(T_v, device=vision.device).unsqueeze(0)
                      >= length.to(vision.device).unsqueeze(1))
            vision = self.vis_enc(vision, src_key_padding_mask=vm_t)

        # ── Cross-attention ─────────────────────────────────────────────
        feats = [t]
        if "audio" in self.task:
            feats.append(self.ca_audio(t, a, key_padding_mask=a_mask))
        if "vit" in self.task:
            T_v  = vision.size(1)
            vm   = (torch.arange(T_v, device=vision.device).unsqueeze(0)
                    >= length.to(vision.device).unsqueeze(1))
            feats.append(self.ca_vision(t, vision, key_padding_mask=vm))

        # ── Fusion ─────────────────────────────────────────────────────
        if self.use_fusion_self_attn:
            fused = self.fusion_sa(torch.stack(feats, dim=1)).mean(dim=1)
        else:
            fused = torch.cat(feats, dim=1)

        # ── Prediction ──────────────────────────────────────────────────
        if self.use_gate_intensity:
            g    = self.gate_head(fused)
            i    = self.intensity_head(fused)
            pred = torch.sigmoid(g) * torch.sigmoid(i)
            return pred, g, i
        else:
            pred = self.pred_head(fused)
            if self.use_sigmoid:
                pred = torch.sigmoid(pred)
            return pred
