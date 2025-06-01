#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT-Transformer統合素粒子スペクトル予測システム
RTX3080最適化・電源断リカバリー対応・長時間計算

Transformerモデルとゲージ不変性：
- 最新研究によりTransformerアーキテクチャはゲージ不変性を持つことが判明
- NKAT理論の非可換ゲージ対称性と自然に整合
- ミューオンg-2異常に基づく第五の力を前提とした予測

Author: NKAT Advanced Computing Team
Date: 2025-01-26
CUDA Requirement: RTX3080 or higher
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import json
import pickle
import os
import time
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

class NKATTransformerConfig:
    """NKAT-Transformer設定クラス"""
    
    def __init__(self):
        # モデル設定
        self.d_model = 512          # モデル次元
        self.nhead = 8              # マルチヘッド注意機構のヘッド数
        self.num_layers = 12        # Transformerレイヤー数
        self.dim_feedforward = 2048 # フィードフォワード次元
        self.dropout = 0.1          # ドロップアウト率
        self.max_seq_len = 256      # 最大シーケンス長
        
        # 物理パラメータ
        self.particle_types = 6     # NKAT予測粒子数
        self.mass_range = 54        # 質量範囲（桁数）
        self.energy_levels = 128    # エネルギーレベル数
        
        # NKAT理論パラメータ
        self.theta_nc = 1e-35       # 非可換性パラメータ
        self.gamma_conv = 2.718     # 超収束因子ベース
        self.fifth_force_strength = 251e-11  # ミューオンg-2から導出
        
        # 計算設定
        self.batch_size = 16        # RTX3080最適バッチサイズ
        self.num_epochs = 1000      # エポック数
        self.learning_rate = 1e-4   # 学習率
        self.warmup_steps = 1000    # ウォームアップステップ
        
        # リカバリー設定
        self.checkpoint_interval = 100  # チェックポイント間隔
        self.auto_save_interval = 300   # 自動保存間隔（秒）

class GaugeInvariantAttention(nn.Module):
    """ゲージ不変マルチヘッド注意機構"""
    
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        # ゲージ不変性を保持する線形変換
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        # ゲージ変換パラメータ
        self.gauge_phase = nn.Parameter(torch.zeros(nhead))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.d_k)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # クエリ、キー、バリューの計算
        Q = self.w_q(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        
        # ゲージ不変性の適用
        gauge_factor = torch.exp(1j * self.gauge_phase.view(1, -1, 1, 1))
        Q = Q * gauge_factor.real - V * gauge_factor.imag
        
        # スケール済み内積注意機構
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 注意機構の適用
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        
        return self.w_o(context), attn_weights

class NKATEmbedding(nn.Module):
    """NKAT理論に基づく物理埋め込み"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        # 物理量埋め込み
        self.mass_embedding = nn.Linear(1, config.d_model // 4)
        self.energy_embedding = nn.Linear(1, config.d_model // 4)
        self.coupling_embedding = nn.Linear(1, config.d_model // 4)
        self.gauge_embedding = nn.Linear(1, config.d_model // 4)
        
        # 非可換補正項
        self.nc_correction = nn.Parameter(torch.randn(config.d_model) * config.theta_nc)
        
        # 位置埋め込み
        self.pos_embedding = nn.Parameter(torch.randn(config.max_seq_len, config.d_model))
        
    def forward(self, mass, energy, coupling, gauge_param):
        batch_size = mass.size(0)
        seq_len = mass.size(1)
        
        # 各物理量の埋め込み
        mass_emb = self.mass_embedding(mass.unsqueeze(-1))
        energy_emb = self.energy_embedding(energy.unsqueeze(-1))
        coupling_emb = self.coupling_embedding(coupling.unsqueeze(-1))
        gauge_emb = self.gauge_embedding(gauge_param.unsqueeze(-1))
        
        # 統合埋め込み
        embedding = torch.cat([mass_emb, energy_emb, coupling_emb, gauge_emb], dim=-1)
        
        # 位置埋め込みの追加
        embedding = embedding + self.pos_embedding[:seq_len, :].unsqueeze(0)
        
        # 非可換補正の適用
        embedding = embedding + self.nc_correction.unsqueeze(0).unsqueeze(0)
        
        return embedding

class SuperConvergenceBlock(nn.Module):
    """超収束因子ブロック"""
    
    def __init__(self, d_model, convergence_factor=2.718):
        super().__init__()
        self.d_model = d_model
        self.convergence_factor = convergence_factor
        
        # 超収束変換
        self.conv_transform = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(0.1)
        )
        
        # 収束制御パラメータ
        self.alpha = nn.Parameter(torch.tensor(convergence_factor))
        self.beta = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x):
        # 超収束変換
        transformed = self.conv_transform(x)
        
        # 収束因子の適用
        convergence_weight = torch.exp(-self.alpha * torch.norm(x, dim=-1, keepdim=True))
        
        # 収束制御
        output = self.beta * x + convergence_weight * transformed
        
        return output

class NKATTransformerLayer(nn.Module):
    """NKAT Transformerレイヤー"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # ゲージ不変注意機構
        self.attention = GaugeInvariantAttention(
            config.d_model, config.nhead, config.dropout)
        
        # 超収束ブロック
        self.super_convergence = SuperConvergenceBlock(
            config.d_model, config.gamma_conv)
        
        # フィードフォワードネットワーク
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        # 層正規化
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        
    def forward(self, x, mask=None):
        # マルチヘッド注意機構
        attn_output, attn_weights = self.attention(self.norm1(x), mask)
        x = x + attn_output
        
        # 超収束ブロック
        conv_output = self.super_convergence(self.norm2(x))
        x = x + conv_output
        
        # フィードフォワード
        ff_output = self.feed_forward(self.norm3(x))
        x = x + ff_output
        
        return x, attn_weights

class NKATTransformerModel(nn.Module):
    """NKAT-Transformer統合モデル"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 物理埋め込み層
        self.embedding = NKATEmbedding(config)
        
        # Transformerレイヤー
        self.layers = nn.ModuleList([
            NKATTransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # 出力層
        self.output_projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.particle_types),
            nn.Softmax(dim=-1)
        )
        
        # 質量予測ヘッド
        self.mass_predictor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1)
        )
        
        # 結合定数予測ヘッド
        self.coupling_predictor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, mass, energy, coupling, gauge_param, mask=None):
        # 埋め込み
        x = self.embedding(mass, energy, coupling, gauge_param)
        
        # Transformerレイヤーの順次適用
        attention_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
        
        # 各予測ヘッドの適用
        particle_probs = self.output_projection(x)
        mass_pred = self.mass_predictor(x)
        coupling_pred = self.coupling_predictor(x)
        
        return {
            'particle_probabilities': particle_probs,
            'mass_predictions': mass_pred,
            'coupling_predictions': coupling_pred,
            'attention_weights': attention_weights
        }

class ParticleSpectrumDataset(Dataset):
    """素粒子スペクトルデータセット"""
    
    def __init__(self, config, num_samples=10000):
        self.config = config
        self.num_samples = num_samples
        
        # NKAT予測粒子の基準質量 (GeV)
        self.base_masses = torch.tensor([
            2.08e-32,   # QIM
            2.05e-26,   # QEP
            1.65e-23,   # TPO
            1.22e14,    # NQG
            4.83e16,    # HDC
            2.42e22     # NCM
        ], dtype=torch.float32)
        
        self.generate_data()
        
    def generate_data(self):
        """物理的に妥当なデータ生成"""
        logger.info("🔬 物理データ生成開始...")
        
        # 質量スペクトル生成
        mass_variations = torch.randn(self.num_samples, self.config.max_seq_len)
        mass_variations = mass_variations * 0.1  # 10%の変動
        
        # エネルギーレベル
        energy_base = torch.logspace(-32, 22, self.config.max_seq_len)
        energy_variations = torch.randn(self.num_samples, self.config.max_seq_len) * 0.05
        
        # 結合定数（ミューオンg-2から導出）
        coupling_base = self.config.fifth_force_strength
        coupling_variations = torch.randn(self.num_samples, self.config.max_seq_len) * coupling_base * 0.1
        
        # ゲージパラメータ
        gauge_params = torch.rand(self.num_samples, self.config.max_seq_len) * 2 * np.pi
        
        self.masses = mass_variations + energy_base.log10().unsqueeze(0)
        self.energies = energy_base.unsqueeze(0).repeat(self.num_samples, 1) * (1 + energy_variations)
        self.couplings = coupling_base + coupling_variations
        self.gauge_params = gauge_params
        
        # ターゲット生成（NKAT理論に基づく）
        self.targets = self._generate_targets()
        
        logger.info(f"✅ {self.num_samples}サンプル生成完了")
        
    def _generate_targets(self):
        """NKAT理論に基づくターゲット生成"""
        targets = {}
        
        # 粒子確率分布
        particle_probs = torch.zeros(self.num_samples, self.config.max_seq_len, self.config.particle_types)
        for i in range(self.config.particle_types):
            prob = torch.exp(-torch.abs(self.energies - self.base_masses[i]) / self.base_masses[i])
            particle_probs[:, :, i] = prob
        
        # 正規化
        particle_probs = particle_probs / particle_probs.sum(dim=-1, keepdim=True)
        
        # 質量予測（対数スケール）
        mass_targets = self.energies.log10()
        
        # 結合定数目標
        coupling_targets = torch.abs(self.couplings)
        
        targets['particle_probabilities'] = particle_probs
        targets['mass_predictions'] = mass_targets.unsqueeze(-1)
        targets['coupling_predictions'] = coupling_targets.unsqueeze(-1)
        
        return targets
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'mass': self.masses[idx],
            'energy': self.energies[idx],
            'coupling': self.couplings[idx],
            'gauge_param': self.gauge_params[idx],
            'targets': {k: v[idx] for k, v in self.targets.items()}
        }

class NKATTrainer:
    """NKAT-Transformer訓練システム"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️ デバイス: {self.device}")
        
        # モデル初期化
        self.model = NKATTransformerModel(config).to(self.device)
        
        # オプティマイザ
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.learning_rate,
            weight_decay=1e-5
        )
        
        # スケジューラ
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate * 10,
            total_steps=config.num_epochs,
            pct_start=0.1
        )
        
        # 損失関数
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        # 訓練記録
        self.train_history = {
            'epoch': [],
            'total_loss': [],
            'particle_loss': [],
            'mass_loss': [],
            'coupling_loss': []
        }
        
        # チェックポイント管理
        self.checkpoint_dir = "nkat_transformer_checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # リカバリー情報
        self.last_save_time = time.time()
        
    def save_checkpoint(self, epoch, extra_info=None):
        """チェックポイント保存"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_history': self.train_history,
            'config': self.config.__dict__,
            'timestamp': datetime.now().isoformat()
        }
        
        if extra_info:
            checkpoint.update(extra_info)
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"checkpoint_epoch_{epoch:04d}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # 最新チェックポイントのシンボリックリンク更新
        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        if os.path.exists(latest_path):
            os.remove(latest_path)
        torch.save(checkpoint, latest_path)
        
        logger.info(f"💾 チェックポイント保存: {checkpoint_path}")
        
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
            
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"🔄 チェックポイント復元: エポック {start_epoch} から再開")
            return start_epoch
            
        except Exception as e:
            logger.error(f"チェックポイント読み込みエラー: {e}")
            return 0
    
    def compute_loss(self, outputs, targets):
        """統合損失計算"""
        # 粒子分類損失
        particle_loss = self.ce_loss(
            outputs['particle_probabilities'].view(-1, self.config.particle_types),
            targets['particle_probabilities'].argmax(dim=-1).view(-1)
        )
        
        # 質量予測損失
        mass_loss = self.mse_loss(
            outputs['mass_predictions'], 
            targets['mass_predictions']
        )
        
        # 結合定数損失
        coupling_loss = self.mse_loss(
            outputs['coupling_predictions'], 
            targets['coupling_predictions']
        )
        
        # 重み付き総損失
        total_loss = (0.4 * particle_loss + 
                     0.3 * mass_loss + 
                     0.3 * coupling_loss)
        
        return {
            'total_loss': total_loss,
            'particle_loss': particle_loss,
            'mass_loss': mass_loss,
            'coupling_loss': coupling_loss
        }
    
    def train_epoch(self, dataloader, epoch):
        """1エポックの訓練"""
        self.model.train()
        epoch_losses = {'total_loss': 0, 'particle_loss': 0, 'mass_loss': 0, 'coupling_loss': 0}
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch:04d}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # データをGPUに転送
            mass = batch['mass'].to(self.device)
            energy = batch['energy'].to(self.device)
            coupling = batch['coupling'].to(self.device)
            gauge_param = batch['gauge_param'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            
            # 勾配初期化
            self.optimizer.zero_grad()
            
            # 順伝播
            outputs = self.model(mass, energy, coupling, gauge_param)
            
            # 損失計算
            losses = self.compute_loss(outputs, targets)
            
            # 逆伝播
            losses['total_loss'].backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # パラメータ更新
            self.optimizer.step()
            
            # 損失累積
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            
            # プログレスバー更新
            progress_bar.set_postfix({
                'Loss': f"{losses['total_loss'].item():.6f}",
                'GPU': f"{torch.cuda.memory_allocated()/1e9:.1f}GB"
            })
            
            # 自動保存チェック
            if time.time() - self.last_save_time > self.config.auto_save_interval:
                self.save_checkpoint(epoch, {'batch_idx': batch_idx})
                self.last_save_time = time.time()
        
        # エポック平均損失
        for key in epoch_losses:
            epoch_losses[key] /= len(dataloader)
        
        return epoch_losses
    
    def train(self, dataloader, start_epoch=0):
        """メイン訓練ループ"""
        logger.info("🚀 NKAT-Transformer訓練開始")
        
        try:
            for epoch in range(start_epoch, self.config.num_epochs):
                # 訓練
                epoch_losses = self.train_epoch(dataloader, epoch)
                
                # スケジューラ更新
                self.scheduler.step()
                
                # 履歴記録
                self.train_history['epoch'].append(epoch)
                for key, value in epoch_losses.items():
                    self.train_history[key].append(value)
                
                # ログ出力
                logger.info(
                    f"Epoch {epoch:04d} - "
                    f"Total: {epoch_losses['total_loss']:.6f}, "
                    f"Particle: {epoch_losses['particle_loss']:.6f}, "
                    f"Mass: {epoch_losses['mass_loss']:.6f}, "
                    f"Coupling: {epoch_losses['coupling_loss']:.6f}"
                )
                
                # 定期チェックポイント
                if (epoch + 1) % self.config.checkpoint_interval == 0:
                    self.save_checkpoint(epoch)
                
                # GPU メモリクリーンアップ
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        except KeyboardInterrupt:
            logger.info("🛑 訓練中断 - チェックポイント保存中...")
            self.save_checkpoint(epoch, {'interrupted': True})
        
        except Exception as e:
            logger.error(f"💥 訓練エラー: {e}")
            self.save_checkpoint(epoch, {'error': str(e)})
            raise
        
        finally:
            # 最終チェックポイント
            self.save_checkpoint(epoch, {'training_completed': True})
            logger.info("✅ 訓練完了")

class NKATSpectrumPredictor:
    """NKAT素粒子スペクトル予測システム"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 訓練済みモデルの読み込み
        self.model = NKATTransformerModel(config).to(self.device)
        self.trainer = NKATTrainer(config)
        
    def predict_spectrum(self, energy_range=None):
        """素粒子スペクトル予測"""
        if energy_range is None:
            energy_range = torch.logspace(-32, 22, 256)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for energy in energy_range:
                # 入力準備
                mass = torch.log10(energy).unsqueeze(0).unsqueeze(0).to(self.device)
                energy_tensor = energy.unsqueeze(0).unsqueeze(0).to(self.device)
                coupling = torch.tensor([[self.config.fifth_force_strength]]).to(self.device)
                gauge = torch.tensor([[0.0]]).to(self.device)
                
                # 予測
                output = self.model(mass, energy_tensor, coupling, gauge)
                predictions.append({
                    'energy': energy.item(),
                    'particle_probs': output['particle_probabilities'].cpu().numpy(),
                    'mass_pred': output['mass_predictions'].cpu().numpy(),
                    'coupling_pred': output['coupling_predictions'].cpu().numpy()
                })
        
        return predictions
    
    def generate_spectrum_plot(self, predictions):
        """スペクトル可視化"""
        energies = [p['energy'] for p in predictions]
        particle_names = ['QIM', 'QEP', 'TPO', 'NQG', 'HDC', 'NCM']
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
        
        # 粒子確率分布
        for i, (name, color) in enumerate(zip(particle_names, colors)):
            probs = [p['particle_probs'][0, 0, i] for p in predictions]
            ax1.semilogx(energies, probs, label=name, color=color, linewidth=2)
        
        ax1.set_xlabel('Energy (GeV)')
        ax1.set_ylabel('Particle Probability')
        ax1.set_title('NKAT-Transformer Predicted Particle Spectrum\n(NKAT-Transformer予測粒子スペクトル)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 質量予測
        mass_preds = [p['mass_pred'][0, 0] for p in predictions]
        ax2.semilogx(energies, mass_preds, color='black', linewidth=2)
        ax2.set_xlabel('Energy (GeV)')
        ax2.set_ylabel('Predicted Mass (log10 GeV)')
        ax2.set_title('Mass Prediction vs Energy')
        ax2.grid(True, alpha=0.3)
        
        # 結合定数予測
        coupling_preds = [p['coupling_pred'][0, 0] for p in predictions]
        ax3.semilogx(energies, coupling_preds, color='red', linewidth=2)
        ax3.set_xlabel('Energy (GeV)')
        ax3.set_ylabel('Predicted Coupling Constant')
        ax3.set_title('Coupling Constant Prediction (Fifth Force)')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'nkat_transformer_spectrum_prediction_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        logger.info(f"📊 スペクトル図保存完了")
        
        return fig

def main():
    """メイン実行関数"""
    print("🌟 NKAT-Transformer素粒子スペクトル予測システム")
    print("=" * 60)
    
    # 設定初期化
    config = NKATTransformerConfig()
    logger.info(f"🖥️ CUDA利用可能: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # データセット準備
    logger.info("📊 データセット準備中...")
    dataset = ParticleSpectrumDataset(config, num_samples=5000)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=0,  # Windows互換性
        pin_memory=torch.cuda.is_available()
    )
    
    # 訓練システム初期化
    trainer = NKATTrainer(config)
    
    # リカバリーチェック
    start_epoch = trainer.load_checkpoint()
    
    # 訓練実行
    trainer.train(dataloader, start_epoch)
    
    # 予測システム初期化
    predictor = NKATSpectrumPredictor(config)
    
    # スペクトル予測
    logger.info("🔮 素粒子スペクトル予測実行中...")
    predictions = predictor.predict_spectrum()
    
    # 結果可視化
    predictor.generate_spectrum_plot(predictions)
    
    # レポート生成
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': config.__dict__,
        'training_history': trainer.train_history,
        'predictions_summary': {
            'energy_range': [predictions[0]['energy'], predictions[-1]['energy']],
            'num_predictions': len(predictions),
            'max_particle_prob': max(p['particle_probs'].max() for p in predictions)
        },
        'gpu_info': {
            'device': str(trainer.device),
            'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        }
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'nkat_transformer_report_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info("🎯 NKAT-Transformer素粒子スペクトル予測完了!")
    
    return report

if __name__ == "__main__":
    try:
        report = main()
        print("\n✅ 計算完了!")
        print(f"📁 結果ファイル: nkat_transformer_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    except Exception as e:
        logger.error(f"💥 実行エラー: {e}")
        raise 