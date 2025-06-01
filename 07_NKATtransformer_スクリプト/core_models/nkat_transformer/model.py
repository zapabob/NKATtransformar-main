# nkat_transformer/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math

from .modules.nkat_attention import NKATAttention, NKATTransformerBlock

class NKATVisionTransformer(nn.Module):
    """
    🚀 NKAT強化版Vision Transformer
    LLMスタイルのハイパーパラメータ対応 + TPE最適化設計
    """
    
    def __init__(self, 
                 img_size: int = 28,
                 patch_size: int = 4, 
                 num_classes: int = 10,
                 embed_dim: int = 512,
                 depth: int = 6,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 # 🌡️ LLMスタイルハイパーパラメータ
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 # 🎯 NKAT理論パラメータ
                 nkat_strength: float = 0.0,
                 nkat_decay: float = 1.0,
                 # 🛡️ 正則化パラメータ
                 dropout_embed: float = 0.1,
                 dropout_attn: float = 0.1,
                 dropout_mlp: float = 0.1,
                 dropout_classifier: float = 0.1,
                 label_smoothing: float = 0.0,
                 # 🔧 追加オプション
                 use_conv_stem: bool = True,
                 use_learnable_pos: bool = True):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size  
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.depth = depth
        
        # ハイパーパラメータ保存（分析用）
        self.hyperparams = {
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'nkat_strength': nkat_strength,
            'nkat_decay': nkat_decay,
            'label_smoothing': label_smoothing
        }
        
        # 🎨 パッチ埋め込み
        if use_conv_stem:
            # ConvStem: より効果的な特徴抽出
            self.patch_embedding = nn.Sequential(
                nn.Conv2d(1, embed_dim // 4, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(embed_dim // 4),
                nn.GELU(),
                nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(embed_dim // 2),
                nn.GELU(),
                nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=patch_size, stride=patch_size),
                nn.BatchNorm2d(embed_dim),
                nn.GELU()
            )
        else:
            # 標準的な線形投影
            self.patch_embedding = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # 🎯 位置埋め込み
        if use_learnable_pos:
            self.pos_embedding = nn.Parameter(
                torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
            )
        else:
            # 固定Sin/Cos位置エンコーディング
            self.register_buffer('pos_embedding', self._get_sincos_pos_embedding())
        
        # 🏷️ クラストークン
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # 📦 入力処理
        self.input_norm = nn.LayerNorm(embed_dim)
        self.input_dropout = nn.Dropout(dropout_embed)
        
        # 🏗️ Transformerブロック群
        self.layers = nn.ModuleList([
            NKATTransformerBlock(
                dim=embed_dim,
                heads=num_heads,
                mlp_ratio=mlp_ratio,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                nkat_strength=nkat_strength,
                nkat_decay=nkat_decay,
                layer_idx=i,
                dropout=dropout_attn
            ) for i in range(depth)
        ])
        
        # 🎯 分類ヘッド
        self.final_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_classifier),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_classifier * 0.5),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(), 
            nn.Dropout(dropout_classifier * 0.5),
            nn.Linear(embed_dim // 4, num_classes)
        )
        
        # 🌡️ 出力温度制御（学習可能）
        self.logits_temperature = nn.Parameter(torch.tensor(temperature))
        
        # ラベルスムージング
        self.label_smoothing = label_smoothing
        
        # 重み初期化
        self.apply(self._init_weights)
    
    def _get_sincos_pos_embedding(self) -> torch.Tensor:
        """Sin/Cos位置エンコーディング生成"""
        pe = torch.zeros(self.num_patches + 1, self.embed_dim)
        position = torch.arange(0, self.num_patches + 1).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * 
                           -(math.log(10000.0) / self.embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def _init_weights(self, m):
        """重み初期化"""
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        🚀 フォワードパス
        
        Args:
            x: 入力画像 (B, C, H, W)
            return_attention: アテンション重みを返すかどうか
        
        Returns:
            分類ロジット（またはアテンション情報付き）
        """
        B = x.shape[0]
        
        # 🎨 パッチ埋め込み
        x = self.patch_embedding(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # 🏷️ クラストークン追加
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)
        
        # 🎯 位置埋め込み追加
        x = x + self.pos_embedding
        x = self.input_norm(x)
        x = self.input_dropout(x)
        
        # 🏗️ Transformerブロック適用
        attention_weights = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # アテンション重みの記録（解析用）
            if return_attention and hasattr(layer.attn, '_last_attn'):
                attention_weights.append(layer.attn._last_attn)
        
        # 🎯 分類
        x = self.final_norm(x)
        cls_output = x[:, 0]  # クラストークンのみ取得
        logits = self.classifier(cls_output)
        
        # 🌡️ 出力温度スケーリング
        logits = logits / self.logits_temperature
        
        if return_attention:
            return logits, attention_weights
        return logits
    
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        📊 損失計算（ラベルスムージング対応）
        
        Args:
            logits: モデル出力
            targets: 正解ラベル
        
        Returns:
            計算された損失
        """
        if self.label_smoothing > 0:
            # ラベルスムージング適用（手動実装でFloat型エラー回避）
            num_classes = logits.size(-1)
            targets = targets.long()  # 確実にLong型にキャスト
            
            # One-hotエンコーディング
            one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
            
            # スムージング適用
            smoothed_targets = one_hot * (1 - self.label_smoothing) + \
                             self.label_smoothing / num_classes
            
            # Log-softmax + NLL損失
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(smoothed_targets * log_probs).sum(dim=-1).mean()
            return loss
        else:
            # 標準CrossEntropy（Long型ターゲット保証）
            targets = targets.long()
            return F.cross_entropy(logits, targets)
    
    def get_nkat_parameters(self) -> Dict[str, Any]:
        """🔍 NKAT関連パラメータの取得"""
        nkat_params = {}
        for name, param in self.named_parameters():
            if any(keyword in name.lower() for keyword in ['nkat', 'alpha', 'beta']):
                nkat_params[name] = param.data.clone()
        return nkat_params
    
    def get_model_info(self) -> Dict[str, Any]:
        """📋 モデル情報取得"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        nkat_params = sum(p.numel() for name, p in self.named_parameters() 
                         if any(keyword in name.lower() for keyword in ['nkat', 'alpha', 'beta']))
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'nkat_parameters': nkat_params,
            'nkat_ratio': nkat_params / max(total_params, 1),
            'hyperparameters': self.hyperparams.copy(),
            'architecture': {
                'embed_dim': self.embed_dim,
                'depth': self.depth,
                'num_patches': self.num_patches,
                'num_classes': self.num_classes
            }
        }


class NKATLightweight(NKATVisionTransformer):
    """🪶 軽量版NKAT（高速実験用）"""
    
    def __init__(self, **kwargs):
        # 軽量化設定のデフォルト値
        lightweight_defaults = {
            'embed_dim': 256,
            'depth': 4,
            'num_heads': 4,
            'mlp_ratio': 2.0,
            'patch_size': 7,  # より大きなパッチサイズ
        }
        
        # デフォルト値を適用
        for key, default_value in lightweight_defaults.items():
            kwargs.setdefault(key, default_value)
        
        super().__init__(**kwargs)


class NKATHeavyweight(NKATVisionTransformer):
    """🏋️ 高性能版NKAT（最終精度重視）"""
    
    def __init__(self, **kwargs):
        # 高性能設定のデフォルト値
        heavyweight_defaults = {
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'mlp_ratio': 4.0,
            'patch_size': 2,  # より細かなパッチサイズ
        }
        
        # デフォルト値を適用
        for key, default_value in heavyweight_defaults.items():
            kwargs.setdefault(key, default_value)
        
        super().__init__(**kwargs) 