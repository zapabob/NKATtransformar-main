import torch
import torch.nn.functional as F
from torch import nn
import math

class NKATAttention(nn.Module):
    """
    🧠 NKAT強化版アテンションメカニズム
    LLMスタイルのTemperature、Top-K、Top-P（Nucleus）サンプリング対応
    動的NKAT補正で層別理論強度調整
    """
    def __init__(self, dim, heads=8, temperature=1.0,
                 top_k=None, top_p=None, nkat_strength=0.,
                 nkat_decay=1.0, layer_idx=0, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.nkat_strength = nkat_strength
        self.nkat_decay = nkat_decay
        self.layer_idx = layer_idx
        self.dim = dim
        self.head_dim = dim // heads

        # Q、K、V投影層
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # 🔥 NKAT理論強化：動的補正パラメータ
        self.nkat_alpha = nn.Parameter(torch.tensor(0.1))  # 学習可能なα
        self.nkat_beta = nn.Parameter(torch.tensor(0.05))  # 学習可能なβ
        
        # 位置エンコーディング強化
        self.register_buffer('pos_bias', torch.zeros(1, heads, 1, 1))

    def apply_temperature_scaling(self, attn_scores):
        """🌡️ Temperature スケーリング適用"""
        return attn_scores / max(self.temperature, 1e-8)

    def apply_top_k_filtering(self, attn_scores):
        """🔝 Top-K フィルタリング"""
        if self.top_k is None or self.top_k <= 0:
            return attn_scores
            
        top_k = min(self.top_k, attn_scores.size(-1))
        val, idx = torch.topk(attn_scores, top_k, dim=-1)
        mask = torch.full_like(attn_scores, float('-inf'))
        return mask.scatter(-1, idx, val)

    def apply_nucleus_filtering(self, attn_scores):
        """🧬 Nucleus (Top-P) フィルタリング"""
        if self.top_p is None or self.top_p >= 1.0:
            return attn_scores
            
        sorted_scores, idx = torch.sort(attn_scores, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_scores, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        
        # Top-Pより大きい確率質量を持つトークンをマスク
        mask = cumulative > self.top_p
        if mask.size(-1) > 1:
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
        
        sorted_scores[mask] = float('-inf')
        return torch.full_like(attn_scores, float('-inf')).scatter(-1, idx, sorted_scores)

    def apply_nkat_enhancement(self, attn, x):
        """
        🎯 動的NKAT理論補正
        層深度に応じて理論強度を調整し、トークン意味情報を統合
        """
        if self.nkat_strength <= 0:
            return attn
            
        # 層依存減衰係数
        layer_coeff = self.nkat_strength * (self.nkat_decay ** self.layer_idx)
        
        # トークンの意味情報抽出（平均 + 分散）
        token_mean = x.mean(dim=-1, keepdim=True)  # (B, N, 1)
        token_var = x.var(dim=-1, keepdim=True, unbiased=False)  # (B, N, 1)
        
        # 🧮 NKAT理論式：θ(x) = α*tanh(μ) + β*sigmoid(σ²)
        theta_semantic = (self.nkat_alpha * torch.tanh(token_mean) + 
                         self.nkat_beta * torch.sigmoid(token_var))
        
        # アテンション重み調整
        nkat_modulation = 1.0 + layer_coeff * theta_semantic.unsqueeze(1)  # (B, 1, N, 1)
        return attn * nkat_modulation

    def forward(self, x):
        """
        🚀 メインフォワードパス
        Temperature → Top-K → Top-P → NKAT補正の順で適用
        """
        B, N, C = x.shape
        
        # Q、K、V計算
        qkv = self.qkv(x).view(B, N, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # (B, N, H, D)
        q = q.transpose(1, 2)  # (B, H, N, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # アテンションスコア計算
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        
        # 位置バイアス追加
        attn_scores = attn_scores + self.pos_bias
        
        # 🌡️ Temperature スケーリング
        attn_scores = self.apply_temperature_scaling(attn_scores)
        
        # 🔝 Top-K フィルタリング
        attn_scores = self.apply_top_k_filtering(attn_scores)
        
        # 🧬 Nucleus (Top-P) フィルタリング  
        attn_scores = self.apply_nucleus_filtering(attn_scores)
        
        # Softmax でアテンション重み計算
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        
        # 🎯 NKAT 動的補正適用
        attn = self.apply_nkat_enhancement(attn, x)
        
        # 値との積算
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out(out)

    def get_attention_entropy(self):
        """📊 アテンション分布のエントロピー計算（解析用）"""
        if hasattr(self, '_last_attn'):
            attn = self._last_attn
            entropy = -torch.sum(attn * torch.log(attn + 1e-8), dim=-1)
            return entropy.mean()
        return torch.tensor(0.)


class NKATTransformerBlock(nn.Module):
    """🏗️ NKAT強化版Transformerブロック"""
    
    def __init__(self, dim, heads, mlp_ratio=4., temperature=1.0,
                 top_k=None, top_p=None, nkat_strength=0.,
                 nkat_decay=1.0, layer_idx=0, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = NKATAttention(
            dim, heads, temperature, top_k, top_p,
            nkat_strength, nkat_decay, layer_idx, dropout
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )
        
        # Skip connection の学習可能重み
        self.skip_alpha = nn.Parameter(torch.tensor(1.0))
        self.skip_beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # Pre-norm + Skip connection with learnable weights
        x = x + self.skip_alpha * self.attn(self.norm1(x))
        x = x + self.skip_beta * self.mlp(self.norm2(x))
        return x 