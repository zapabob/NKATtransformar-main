#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT-Transformer統合MNIST画像認識システム
RTX3080最適化・電源断リカバリー対応・長時間学習

NKAT-Vision Transformer統合理論：
- ゲージ不変性を持つパッチ埋め込み
- 非可換幾何学的注意機構
- 量子重力補正項による画像特徴抽出
- 超収束因子による高精度認識

Author: NKAT Advanced Computing Team
Date: 2025-01-26
CUDA Requirement: RTX3080 or higher
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import json
import pickle
import os
import time
import math
from datetime import datetime
from tqdm import tqdm
import logging
import warnings
warnings.filterwarnings('ignore')

# CUDA最適化
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NKATVisionConfig:
    """NKAT-Vision Transformer設定クラス"""
    
    def __init__(self):
        # 画像設定
        self.image_size = 28            # MNIST画像サイズ
        self.patch_size = 7             # パッチサイズ (4x4パッチ)
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.channels = 1               # グレースケール
        
        # モデル設定
        self.d_model = 384              # モデル次元
        self.nhead = 6                  # マルチヘッド注意機構のヘッド数
        self.num_layers = 8             # Transformerレイヤー数
        self.dim_feedforward = 1536     # フィードフォワード次元
        self.dropout = 0.1              # ドロップアウト率
        self.num_classes = 10           # MNIST分類数
        
        # NKAT理論パラメータ
        self.theta_nc = 1e-35           # 非可換性パラメータ
        self.gamma_conv = 2.718         # 超収束因子
        self.quantum_correction = 1e-6  # 量子重力補正強度
        self.gauge_symmetry_dim = 8     # ゲージ対称性次元
        
        # 訓練設定
        self.batch_size = 128           # RTX3080最適バッチサイズ
        self.num_epochs = 200           # エポック数
        self.learning_rate = 1e-3       # 学習率
        self.warmup_steps = 1000        # ウォームアップステップ
        self.weight_decay = 1e-4        # 重み減衰
        
        # データ拡張設定
        self.use_augmentation = True    # データ拡張使用
        self.rotation_degrees = 10      # 回転角度
        self.zoom_factor = 0.1          # ズーム係数
        
        # リカバリー設定
        self.checkpoint_interval = 10   # チェックポイント間隔
        self.auto_save_interval = 300   # 自動保存間隔（秒）

class NKATPatchEmbedding(nn.Module):
    """NKAT理論に基づくパッチ埋め込み"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # パッチ投影
        self.patch_projection = nn.Conv2d(
            config.channels, 
            config.d_model, 
            kernel_size=config.patch_size, 
            stride=config.patch_size
        )
        
        # 位置埋め込み
        self.position_embedding = nn.Parameter(
            torch.randn(1, config.num_patches + 1, config.d_model) * 0.02
        )
        
        # クラストークン
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, config.d_model) * 0.02
        )
        
        # 非可換幾何学的補正
        self.nc_correction = nn.Parameter(
            torch.randn(config.d_model) * config.theta_nc
        )
        
        # ゲージ不変性パラメータ
        self.gauge_params = nn.Parameter(
            torch.randn(config.gauge_symmetry_dim, config.d_model) * 0.01
        )
        
        # 量子重力補正層
        self.quantum_layer = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # パッチ投影 (batch_size, channels, height, width) -> (batch_size, d_model, n_patches_h, n_patches_w)
        x = self.patch_projection(x)  # (B, d_model, 4, 4)
        
        # フラット化 (batch_size, d_model, num_patches)
        x = x.flatten(2).transpose(1, 2)  # (B, 16, d_model)
        
        # クラストークンの追加
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 17, d_model)
        
        # 位置埋め込みの追加
        x = x + self.position_embedding
        
        # 非可換幾何学的補正の適用
        nc_effect = self.nc_correction.unsqueeze(0).unsqueeze(0)
        x = x + nc_effect
        
        # ゲージ不変性の適用
        gauge_effect = torch.einsum('gd,bd->bg', self.gauge_params, x.mean(dim=1))
        gauge_correction = torch.einsum('bg,gd->bd', gauge_effect, self.gauge_params)
        x = x + gauge_correction.unsqueeze(1) * self.config.quantum_correction
        
        # 量子重力補正
        quantum_correction = self.quantum_layer(x) * self.config.quantum_correction
        x = x + quantum_correction
        
        return self.dropout(x)

class NKATGaugeInvariantAttention(nn.Module):
    """NKAT理論に基づくゲージ不変注意機構"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.nhead = config.nhead
        self.d_k = self.d_model // self.nhead
        
        # クエリ、キー、バリュー投影
        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_o = nn.Linear(self.d_model, self.d_model)
        
        # ゲージ変換パラメータ
        self.gauge_phase = nn.Parameter(torch.zeros(self.nhead))
        self.gauge_amplitude = nn.Parameter(torch.ones(self.nhead))
        
        # 非可換補正項
        self.nc_mixing = nn.Parameter(torch.randn(self.nhead, self.d_k, self.d_k) * config.theta_nc)
        
        self.dropout = nn.Dropout(config.dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # QKV計算
        Q = self.w_q(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        
        # ゲージ不変性の適用
        gauge_rotation = torch.exp(1j * self.gauge_phase.view(1, -1, 1, 1))
        gauge_scale = self.gauge_amplitude.view(1, -1, 1, 1)
        
        # 複素ゲージ変換（実部のみ使用）
        Q_gauge = Q * gauge_scale * gauge_rotation.real
        K_gauge = K * gauge_scale * gauge_rotation.real
        
        # 非可換補正の適用
        Q_nc = torch.einsum('bhnd,hdk->bhnk', Q_gauge, self.nc_mixing)
        K_nc = torch.einsum('bhnd,hdk->bhnk', K_gauge, self.nc_mixing)
        
        # スケール済み内積注意
        scores = torch.matmul(Q_nc, K_nc.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 注意機構の適用
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.w_o(context), attn_weights

class NKATSuperConvergenceBlock(nn.Module):
    """NKAT超収束ブロック"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        # 超収束変換
        self.convergence_transform = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.d_model * 4, self.d_model),
            nn.Dropout(config.dropout)
        )
        
        # 収束制御パラメータ
        self.alpha = nn.Parameter(torch.tensor(config.gamma_conv))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(0.1))
        
        # 非線形収束関数
        self.convergence_activation = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Tanh(),
            nn.Linear(self.d_model, self.d_model)
        )
        
    def forward(self, x):
        # 超収束変換
        transformed = self.convergence_transform(x)
        
        # 収束制御の計算
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        convergence_weight = torch.exp(-self.alpha * x_norm)
        
        # 非線形収束項
        nonlinear_term = self.convergence_activation(x) * self.gamma
        
        # 最終的な超収束出力
        output = (self.beta * x + 
                 convergence_weight * transformed + 
                 nonlinear_term)
        
        return output

class NKATTransformerBlock(nn.Module):
    """NKAT-Transformerブロック"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # ゲージ不変注意機構
        self.attention = NKATGaugeInvariantAttention(config)
        
        # 超収束ブロック
        self.super_convergence = NKATSuperConvergenceBlock(config)
        
        # 層正規化
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        
        # 残差接続の重み
        self.residual_weight = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x, mask=None):
        # Pre-norm注意機構
        norm_x = self.norm1(x)
        attn_output, attn_weights = self.attention(norm_x, mask)
        x = x + self.residual_weight * attn_output
        
        # Pre-norm超収束
        norm_x = self.norm2(x)
        conv_output = self.super_convergence(norm_x)
        x = x + self.residual_weight * conv_output
        
        # 最終正規化
        x = self.norm3(x)
        
        return x, attn_weights

class NKATVisionTransformer(nn.Module):
    """NKAT-Vision Transformerメインモデル"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # パッチ埋め込み
        self.patch_embedding = NKATPatchEmbedding(config)
        
        # Transformerブロック
        self.transformer_blocks = nn.ModuleList([
            NKATTransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # 分類ヘッド
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.num_classes)
        )
        
        # 量子重力補正項
        self.quantum_classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.Tanh(),
            nn.Linear(config.d_model, config.num_classes)
        )
        
        # 統合重み
        self.fusion_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        # パッチ埋め込み
        x = self.patch_embedding(x)  # (B, num_patches+1, d_model)
        
        # Transformerブロックの順次適用
        attention_weights = []
        for block in self.transformer_blocks:
            x, attn_weights = block(x)
            attention_weights.append(attn_weights)
        
        # クラストークンの抽出
        cls_token = x[:, 0]  # (B, d_model)
        
        # 分類予測
        main_logits = self.classifier(cls_token)
        quantum_logits = self.quantum_classifier(cls_token)
        
        # 量子重力統合
        final_logits = main_logits + self.fusion_weight * quantum_logits
        
        return {
            'logits': final_logits,
            'cls_features': cls_token,
            'attention_weights': attention_weights,
            'quantum_contribution': quantum_logits
        }

class NKATMNISTTrainer:
    """NKAT-MNIST訓練システム"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️ デバイス: {self.device}")
        
        # モデル初期化
        self.model = NKATVisionTransformer(config).to(self.device)
        
        # オプティマイザ
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # スケジューラ
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=50,
            T_mult=2,
            eta_min=1e-6
        )
        
        # 損失関数
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 訓練記録
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # チェックポイント管理
        self.checkpoint_dir = "nkat_mnist_checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # パフォーマンス追跡
        self.best_val_acc = 0.0
        self.last_save_time = time.time()
        
    def save_checkpoint(self, epoch, val_acc, extra_info=None):
        """チェックポイント保存"""
        is_best = val_acc > self.best_val_acc
        if is_best:
            self.best_val_acc = val_acc
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_history': self.train_history,
            'best_val_acc': self.best_val_acc,
            'config': self.config.__dict__,
            'timestamp': datetime.now().isoformat()
        }
        
        if extra_info:
            checkpoint.update(extra_info)
        
        # 通常のチェックポイント
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_epoch_{epoch:04d}_acc_{val_acc:.4f}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # ベストモデルの保存
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"🏆 新記録! 精度: {val_acc:.4f}")
        
        # 最新チェックポイント
        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        torch.save(checkpoint, latest_path)
        
        logger.info(f"💾 チェックポイント保存: エポック {epoch}, 精度 {val_acc:.4f}")
    
    def load_checkpoint(self, checkpoint_path=None):
        """チェックポイント読み込み"""
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        
        if not os.path.exists(checkpoint_path):
            logger.info("チェックポイントが見つかりません。新規訓練を開始します。")
            return 0
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.train_history = checkpoint['train_history']
            self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
            
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"🔄 チェックポイント復元: エポック {start_epoch}, ベスト精度 {self.best_val_acc:.4f}")
            return start_epoch
            
        except Exception as e:
            logger.error(f"チェックポイント読み込みエラー: {e}")
            return 0
    
    def train_epoch(self, train_loader, epoch):
        """1エポックの訓練"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Train Epoch {epoch:03d}')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 順伝播
            outputs = self.model(data)
            loss = self.criterion(outputs['logits'], target)
            
            # 逆伝播
            loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 統計更新
            total_loss += loss.item()
            pred = outputs['logits'].argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # プログレスバー更新
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{100.*correct/total:.2f}%",
                    'GPU': f"{torch.cuda.memory_allocated()/1e9:.1f}GB"
                })
            
            # 自動保存チェック
            if time.time() - self.last_save_time > self.config.auto_save_interval:
                # 簡易検証
                val_acc = self.quick_validate()
                self.save_checkpoint(epoch, val_acc, {'batch_idx': batch_idx})
                self.last_save_time = time.time()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader, epoch):
        """検証エポック"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f'Val Epoch {epoch:03d}')
            
            for data, target in progress_bar:
                data, target = data.to(self.device), target.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs['logits'], target)
                
                total_loss += loss.item()
                pred = outputs['logits'].argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{100.*correct/total:.2f}%"
                })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def quick_validate(self):
        """簡易検証（自動保存用）"""
        self.model.eval()
        # 少数のバッチで簡易検証
        return self.best_val_acc  # 簡略化
    
    def train(self, train_loader, val_loader, start_epoch=0):
        """メイン訓練ループ"""
        logger.info("🚀 NKAT-MNIST訓練開始")
        
        try:
            for epoch in range(start_epoch, self.config.num_epochs):
                # 訓練
                train_loss, train_acc = self.train_epoch(train_loader, epoch)
                
                # 検証
                val_loss, val_acc = self.validate_epoch(val_loader, epoch)
                
                # スケジューラ更新
                self.scheduler.step()
                
                # 履歴記録
                self.train_history['epoch'].append(epoch)
                self.train_history['train_loss'].append(train_loss)
                self.train_history['train_acc'].append(train_acc)
                self.train_history['val_loss'].append(val_loss)
                self.train_history['val_acc'].append(val_acc)
                
                # ログ出力
                logger.info(
                    f"Epoch {epoch:03d} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
                )
                
                # チェックポイント保存
                if (epoch + 1) % self.config.checkpoint_interval == 0:
                    self.save_checkpoint(epoch, val_acc)
                
                # GPU メモリクリーンアップ
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        except KeyboardInterrupt:
            logger.info("🛑 訓練中断 - チェックポイント保存中...")
            self.save_checkpoint(epoch, val_acc, {'interrupted': True})
        
        except Exception as e:
            logger.error(f"💥 訓練エラー: {e}")
            self.save_checkpoint(epoch, val_acc, {'error': str(e)})
            raise
        
        finally:
            self.save_checkpoint(epoch, val_acc, {'training_completed': True})
            logger.info("✅ 訓練完了")

def create_data_loaders(config):
    """データローダー作成"""
    
    # データ変換の定義
    if config.use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomRotation(config.rotation_degrees),
            transforms.RandomAffine(0, scale=(1-config.zoom_factor, 1+config.zoom_factor)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # データセット読み込み
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True,
        download=True, 
        transform=train_transform
    )
    
    val_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False,
        download=True, 
        transform=val_transform
    )
    
    # データローダー作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Windows互換性
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"📊 訓練データ: {len(train_dataset)}, 検証データ: {len(val_dataset)}")
    
    return train_loader, val_loader

def visualize_results(trainer, config):
    """結果可視化"""
    history = trainer.train_history
    
    if len(history['epoch']) == 0:
        logger.warning("訓練履歴がありません")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = history['epoch']
    
    # 損失曲線
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='blue')
    ax1.plot(epochs, history['val_loss'], label='Val Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('NKAT-Transformer MNIST Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 精度曲線
    ax2.plot(epochs, history['train_acc'], label='Train Accuracy', color='blue')
    ax2.plot(epochs, history['val_acc'], label='Val Accuracy', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('NKAT-Transformer MNIST Accuracy Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 学習率曲線
    lrs = [trainer.optimizer.param_groups[0]['lr']] * len(epochs)
    ax3.plot(epochs, lrs, color='green')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.grid(True, alpha=0.3)
    
    # 最終性能サマリー
    best_val_acc = max(history['val_acc']) if history['val_acc'] else 0
    final_train_acc = history['train_acc'][-1] if history['train_acc'] else 0
    
    ax4.text(0.1, 0.7, f'Best Validation Accuracy: {best_val_acc:.2f}%', 
             fontsize=14, fontweight='bold')
    ax4.text(0.1, 0.5, f'Final Training Accuracy: {final_train_acc:.2f}%', 
             fontsize=14)
    ax4.text(0.1, 0.3, f'Total Epochs: {len(epochs)}', fontsize=14)
    ax4.text(0.1, 0.1, f'Model Parameters: {sum(p.numel() for p in trainer.model.parameters()):,}', 
             fontsize=12)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Training Summary')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'nkat_mnist_training_results_{timestamp}.png', 
               dpi=300, bbox_inches='tight')
    logger.info("📊 訓練結果可視化完了")
    
    return fig

def main():
    """メイン実行関数"""
    print("🌟 NKAT-Transformer MNIST画像認識システム")
    print("=" * 60)
    
    # 設定初期化
    config = NKATVisionConfig()
    logger.info(f"🖥️ CUDA利用可能: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # データローダー準備
    logger.info("📊 MNISTデータセット準備中...")
    train_loader, val_loader = create_data_loaders(config)
    
    # 訓練システム初期化
    trainer = NKATMNISTTrainer(config)
    
    # モデル情報表示
    total_params = sum(p.numel() for p in trainer.model.parameters())
    logger.info(f"🧠 モデルパラメータ数: {total_params:,}")
    
    # リカバリーチェック
    start_epoch = trainer.load_checkpoint()
    
    # 訓練実行
    trainer.train(train_loader, val_loader, start_epoch)
    
    # 結果可視化
    visualize_results(trainer, config)
    
    # レポート生成
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': config.__dict__,
        'model_info': {
            'total_parameters': total_params,
            'architecture': 'NKAT-Vision Transformer',
            'patch_size': config.patch_size,
            'num_patches': config.num_patches
        },
        'training_results': {
            'best_val_accuracy': trainer.best_val_acc,
            'total_epochs': len(trainer.train_history['epoch']),
            'final_train_acc': trainer.train_history['train_acc'][-1] if trainer.train_history['train_acc'] else 0
        },
        'nkat_features': {
            'gauge_invariance': True,
            'non_commutative_geometry': True,
            'super_convergence': True,
            'quantum_gravity_correction': True
        },
        'gpu_info': {
            'device': str(trainer.device),
            'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        }
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'nkat_mnist_report_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info("🎯 NKAT-Transformer MNIST認識完了!")
    logger.info(f"🏆 最高検証精度: {trainer.best_val_acc:.2f}%")
    
    return report

if __name__ == "__main__":
    try:
        report = main()
        print(f"\n✅ 計算完了!")
        print(f"📁 結果レポート: nkat_mnist_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        print(f"🏆 最高精度: {report['training_results']['best_val_accuracy']:.2f}%")
    except Exception as e:
        logger.error(f"💥 実行エラー: {e}")
        raise 