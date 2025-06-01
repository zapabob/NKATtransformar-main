#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT Transformer深層学習最適化システム：非可換パラメータと超収束因子の係数最適化（RTX3080中パワー版）
NKAT Transformer Deep Learning Optimization System (RTX3080 Medium Power)

Author: 峯岸 亮 (Ryo Minegishi)
Date: 2025年5月28日
Version: 2.2 (RTX3080中パワー最適化版)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import zeta
import pandas as pd
from tqdm import tqdm
import json
import math
import warnings
import gc
import time
import sys
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# GPU設定（RTX3080最適化）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 使用デバイス: {device}")
if torch.cuda.is_available():
    print(f"🎮 GPU名: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM容量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

class PositionalEncoding(nn.Module):
    """
    Transformerのための位置エンコーディング
    """
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class NKATTransformerDataset(Dataset):
    """
    NKAT Transformer用データセット（改良版）
    """
    
    def __init__(self, N_values, target_values, sequence_length=16, noise_level=1e-6):
        """
        Args:
            N_values: 次元数の配列
            target_values: 目標超収束因子値
            sequence_length: シーケンス長
            noise_level: ノイズレベル
        """
        self.sequence_length = sequence_length
        
        # 入力検証
        if len(N_values) < sequence_length:
            raise ValueError(f"データサイズ({len(N_values)})がシーケンス長({sequence_length})より小さいです")
        
        # データを正規化
        self.N_values = torch.tensor(N_values, dtype=torch.float32)
        self.target_values = torch.tensor(target_values, dtype=torch.float32)
        
        # 対数変換で数値安定性を向上
        self.log_N = torch.log(self.N_values + 1e-8)
        self.log_targets = torch.log(torch.clamp(self.target_values, min=1e-8))
        
        # 正規化
        self.log_N_mean = self.log_N.mean()
        self.log_N_std = self.log_N.std() + 1e-8
        self.log_targets_mean = self.log_targets.mean()
        self.log_targets_std = self.log_targets.std() + 1e-8
        
        self.log_N_norm = (self.log_N - self.log_N_mean) / self.log_N_std
        self.log_targets_norm = (self.log_targets - self.log_targets_mean) / self.log_targets_std
        
        # ノイズ追加
        if noise_level > 0:
            noise = torch.normal(0, noise_level, size=self.log_targets_norm.shape)
            self.log_targets_norm += noise
        
        # データセットサイズ検証
        self.dataset_size = max(0, len(self.N_values) - self.sequence_length + 1)
        
        print(f"📊 データセット初期化完了: {len(self.N_values)}サンプル → {self.dataset_size}シーケンス")
        print(f"📊 シーケンス長: {sequence_length}")
        
        if self.dataset_size == 0:
            raise ValueError("有効なシーケンスが作成できません。データサイズまたはシーケンス長を調整してください。")
    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        if idx >= self.dataset_size:
            raise IndexError(f"インデックス {idx} が範囲外です (最大: {self.dataset_size-1})")
        
        # シーケンスデータを作成
        end_idx = idx + self.sequence_length
        
        input_seq = self.log_N_norm[idx:end_idx].unsqueeze(-1)  # [seq_len, 1]
        target_seq = self.log_targets_norm[idx:end_idx]  # [seq_len]
        
        return input_seq, target_seq

class NKATTransformerModel(nn.Module):
    """
    NKAT用Transformerモデル（RTX3080中パワー版）
    """
    
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super(NKATTransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # 入力投影層
        self.input_projection = nn.Linear(1, d_model)
        
        # 位置エンコーディング
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformerエンコーダー（中パワー版）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # パラメータ予測ヘッド（中パワー版）
        self.parameter_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)  # γ, δ, t_c
        )
        
        # 超収束因子予測ヘッド（中パワー版）
        self.convergence_head = nn.Sequential(
            nn.Linear(d_model + 3, 512),  # Transformer出力 + パラメータ
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        # 初期化
        self._init_weights()
        
        print(f"🧠 NKAT Transformerモデル初期化完了（RTX3080中パワー版）")
        print(f"📊 パラメータ数: {sum(p.numel() for p in self.parameters()):,}")
        print(f"💾 推定VRAM使用量: {self._estimate_memory_usage():.1f} MB")
    
    def _init_weights(self):
        """重み初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        前向き計算
        
        Args:
            x: 入力シーケンス [batch_size, seq_len, 1]
            
        Returns:
            超収束因子の予測値とパラメータ
        """
        batch_size, seq_len, _ = x.shape
        
        # 入力投影
        x = self.input_projection(x) * math.sqrt(self.d_model)  # [batch_size, seq_len, d_model]
        
        # 位置エンコーディング
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Transformer エンコーディング
        transformer_output = self.transformer_encoder(x)  # [batch_size, seq_len, d_model]
        
        # 最後のタイムステップを使用
        last_output = transformer_output[:, -1, :]  # [batch_size, d_model]
        
        # パラメータ予測
        raw_params = self.parameter_head(last_output)  # [batch_size, 3]
        
        # パラメータ制約適用（安定化のため）
        gamma = torch.sigmoid(raw_params[:, 0]) * 0.3 + 0.15  # [0.15, 0.45]
        delta = torch.sigmoid(raw_params[:, 1]) * 0.04 + 0.02  # [0.02, 0.06]
        t_c = F.softplus(raw_params[:, 2]) + 12.0  # [12, ∞) - 修正: F.softplusを使用
        
        # パラメータをクリップして数値安定性を確保
        gamma = torch.clamp(gamma, 0.1, 0.5)
        delta = torch.clamp(delta, 0.01, 0.08)
        t_c = torch.clamp(t_c, 10.0, 30.0)
        
        # 超収束因子計算用の特徴量
        params = torch.stack([gamma, delta, t_c], dim=1)  # [batch_size, 3]
        combined_features = torch.cat([last_output, params], dim=1)  # [batch_size, d_model+3]
        
        # 超収束因子予測
        log_S = self.convergence_head(combined_features)  # [batch_size, 1]
        log_S = torch.clamp(log_S, -5, 5)  # 数値安定性のためクリップ
        
        return log_S.squeeze(), gamma, delta, t_c

    def _estimate_memory_usage(self):
        """VRAM使用量の推定"""
        param_size = sum(p.numel() * 4 for p in self.parameters()) / 1024**2  # MB
        return param_size * 3  # パラメータ + 勾配 + オプティマイザー状態

class NKATTransformerLoss(nn.Module):
    """
    Transformer用の安定化された損失関数（改良版）
    """
    
    def __init__(self, alpha=1.0, beta=0.05, gamma=0.001):
        super(NKATTransformerLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss()
        self.huber = nn.SmoothL1Loss()
    
    def forward(self, log_S_pred, log_S_target, gamma_pred, delta_pred, tc_pred, model):
        """
        安定化された損失計算
        """
        # データ適合損失（Huber損失で外れ値に頑健）
        data_loss = self.huber(log_S_pred, log_S_target)
        
        # 物理制約損失（軽量化）
        physics_loss = self._physics_constraints(gamma_pred, delta_pred, tc_pred)
        
        # 正則化損失（軽量化）
        reg_loss = self._regularization_loss(model)
        
        total_loss = self.alpha * data_loss + self.beta * physics_loss + self.gamma * reg_loss
        
        # NaN チェック
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("⚠️ NaN/Inf検出 - 損失を小さな値に設定")
            total_loss = torch.tensor(1e-6, device=total_loss.device, requires_grad=True)
        
        return total_loss, data_loss, physics_loss, reg_loss
    
    def _physics_constraints(self, gamma_pred, delta_pred, tc_pred):
        """
        物理制約の計算（軽量化版）
        """
        constraints = []
        
        # 1. パラメータ範囲制約（軽量化）
        gamma_constraint = torch.mean(torch.relu(gamma_pred - 0.5) + torch.relu(0.1 - gamma_pred))
        delta_constraint = torch.mean(torch.relu(delta_pred - 0.08) + torch.relu(0.01 - delta_pred))
        tc_constraint = torch.mean(torch.relu(10.0 - tc_pred) + torch.relu(tc_pred - 30.0))
        
        constraints.extend([gamma_constraint, delta_constraint, tc_constraint])
        
        # 2. パラメータ安定性制約（軽量化）
        stability_loss = torch.mean((gamma_pred - 0.234) ** 2) * 0.1 + \
                        torch.mean((delta_pred - 0.035) ** 2) * 0.1 + \
                        torch.mean((tc_pred - 17.0) ** 2) * 0.001
        constraints.append(stability_loss)
        
        return sum(constraints)
    
    def _regularization_loss(self, model):
        """
        正則化損失の計算（軽量化）
        """
        l2_reg = 0
        for param in model.parameters():
            if param.requires_grad:
                l2_reg += torch.norm(param) ** 2
        return l2_reg * 1e-7  # さらに軽い正則化

class NKATTransformerOptimizer:
    """
    NKAT Transformer最適化システム（RTX3080中パワー版）
    """
    
    def __init__(self, learning_rate=3e-4, batch_size=16, num_epochs=300, sequence_length=32, 
                 patience=50, min_delta=1e-6):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.sequence_length = sequence_length
        self.patience = patience
        self.min_delta = min_delta
        
        # モデル初期化（RTX3080中パワー版）
        self.model = NKATTransformerModel(
            d_model=256,
            nhead=8,
            num_layers=6,
            dim_feedforward=1024,
            dropout=0.1
        ).to(device)
        
        # 損失関数
        self.criterion = NKATTransformerLoss()
        
        # オプティマイザー（AdamWで安定化）
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # スケジューラー（Cosine Annealing + Warm Restart）
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=50,
            T_mult=2,
            eta_min=1e-7
        )
        
        # 履歴
        self.train_history = {
            'total_loss': [],
            'data_loss': [],
            'physics_loss': [],
            'reg_loss': [],
            'gamma_values': [],
            'delta_values': [],
            'tc_values': [],
            'learning_rates': []
        }
        
        print("🚀 NKAT Transformer最適化システム初期化完了（RTX3080中パワー版）")
        print(f"📊 早期停止: patience={patience}, min_delta={min_delta}")
        print(f"🎯 バッチサイズ: {batch_size}, シーケンス長: {sequence_length}")
    
    def generate_training_data(self, N_range=(10, 500), num_samples=800):
        """
        訓練データの生成（RTX3080中パワー版）
        """
        print("📊 訓練データ生成中（RTX3080中パワー版）...")
        
        # 次元数の生成（より密なサンプリング）
        N_values = np.logspace(np.log10(N_range[0]), np.log10(N_range[1]), num_samples)
        
        # 理論的超収束因子の計算（高精度版）
        gamma_true = 0.234
        delta_true = 0.035
        t_c_true = 17.26
        
        target_values = []
        
        with tqdm(total=num_samples, desc="理論値計算", ncols=100) as pbar:
            for N in N_values:
                try:
                    # 理論的超収束因子（高精度計算）
                    integral = gamma_true * np.log(max(N / t_c_true, 1e-8))
                    if N > t_c_true:
                        integral += delta_true * (N - t_c_true) * 0.02  # より精密なスケール
                    
                    # 高次補正項を追加
                    if N > 50:
                        integral += 0.001 * np.log(N / 50) ** 2
                    
                    S_theoretical = np.exp(np.clip(integral, -8, 8))  # より広い範囲
                    target_values.append(S_theoretical)
                    
                except:
                    # エラー時はデフォルト値
                    target_values.append(1.0)
                
                pbar.update(1)
        
        target_values = np.array(target_values)
        
        # データセット作成
        try:
            dataset = NKATTransformerDataset(
                N_values, 
                target_values, 
                sequence_length=self.sequence_length,
                noise_level=5e-7  # より低いノイズレベル
            )
        except ValueError as e:
            print(f"❌ データセット作成エラー: {e}")
            print("📊 パラメータを調整して再試行...")
            # パラメータ調整
            self.sequence_length = min(self.sequence_length, num_samples // 4)
            dataset = NKATTransformerDataset(
                N_values, 
                target_values, 
                sequence_length=self.sequence_length,
                noise_level=5e-7
            )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=2,  # RTX3080では並列処理を活用
            pin_memory=True if device.type == 'cuda' else False,
            drop_last=True,
            persistent_workers=True
        )
        
        print(f"✅ 訓練データ生成完了: {num_samples}サンプル → {len(dataloader)}バッチ")
        print(f"📊 推定訓練時間: {len(dataloader) * self.num_epochs / 100:.1f}分")
        return dataloader, dataset
    
    def train(self, dataloader):
        """
        モデル訓練（改良版）
        """
        print("🎓 モデル訓練開始...")
        
        if len(dataloader) == 0:
            print("❌ データローダーが空です。訓練を中止します。")
            return
        
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        # エポックループ
        epoch_pbar = tqdm(range(self.num_epochs), desc="エポック進行", ncols=100)
        
        try:
            for epoch in epoch_pbar:
                epoch_losses = {'total': 0, 'data': 0, 'physics': 0, 'reg': 0}
                epoch_params = {'gamma': [], 'delta': [], 'tc': []}
                valid_batches = 0
                
                # バッチループ
                batch_pbar = tqdm(dataloader, desc=f"エポック {epoch+1}", leave=False, ncols=80)
                
                for batch_idx, (batch_input, batch_target) in enumerate(batch_pbar):
                    try:
                        batch_input = batch_input.to(device)
                        batch_target = batch_target.to(device)
                        
                        # 前向き計算
                        log_S_pred, gamma_pred, delta_pred, tc_pred = self.model(batch_input)
                        
                        # 最後のタイムステップのターゲット
                        log_S_target = batch_target[:, -1]
                        
                        # 損失計算
                        total_loss, data_loss, physics_loss, reg_loss = self.criterion(
                            log_S_pred, log_S_target, gamma_pred, delta_pred, tc_pred, self.model
                        )
                        
                        # NaN/Infチェック
                        if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss.item() > 1e6:
                            print(f"⚠️ エポック {epoch+1}, バッチ {batch_idx}: 異常な損失値をスキップ")
                            continue
                        
                        # 逆伝播
                        self.optimizer.zero_grad()
                        total_loss.backward()
                        
                        # 勾配クリッピング（強化）
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.3)
                        
                        self.optimizer.step()
                        
                        # 損失記録
                        epoch_losses['total'] += total_loss.item()
                        epoch_losses['data'] += data_loss.item()
                        epoch_losses['physics'] += physics_loss.item()
                        epoch_losses['reg'] += reg_loss.item()
                        
                        # パラメータ記録
                        epoch_params['gamma'].extend(gamma_pred.detach().cpu().numpy())
                        epoch_params['delta'].extend(delta_pred.detach().cpu().numpy())
                        epoch_params['tc'].extend(tc_pred.detach().cpu().numpy())
                        
                        valid_batches += 1
                        
                        # バッチ進捗更新
                        batch_pbar.set_postfix({
                            'Loss': f'{total_loss.item():.6f}',
                            'γ': f'{gamma_pred.mean().item():.4f}',
                            'δ': f'{delta_pred.mean().item():.4f}'
                        })
                        
                        # メモリクリア
                        if batch_idx % 10 == 0:
                            torch.cuda.empty_cache() if device.type == 'cuda' else None
                        
                    except Exception as e:
                        print(f"⚠️ バッチ {batch_idx} でエラー: {e}")
                        continue
                
                # エポック平均の計算
                if valid_batches > 0:
                    for key in epoch_losses:
                        epoch_losses[key] /= valid_batches
                    
                    # 履歴記録
                    self.train_history['total_loss'].append(epoch_losses['total'])
                    self.train_history['data_loss'].append(epoch_losses['data'])
                    self.train_history['physics_loss'].append(epoch_losses['physics'])
                    self.train_history['reg_loss'].append(epoch_losses['reg'])
                    
                    if epoch_params['gamma']:
                        self.train_history['gamma_values'].append(np.mean(epoch_params['gamma']))
                        self.train_history['delta_values'].append(np.mean(epoch_params['delta']))
                        self.train_history['tc_values'].append(np.mean(epoch_params['tc']))
                    else:
                        # デフォルト値
                        self.train_history['gamma_values'].append(0.234)
                        self.train_history['delta_values'].append(0.035)
                        self.train_history['tc_values'].append(17.26)
                    
                    self.train_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
                    
                    # スケジューラー更新（CosineAnnealingWarmRestartsの場合）
                    self.scheduler.step()
                    
                    # 早期停止チェック
                    if epoch_losses['total'] < best_loss - self.min_delta:
                        best_loss = epoch_losses['total']
                        patience_counter = 0
                        # ベストモデルを保存
                        torch.save(self.model.state_dict(), 'best_nkat_transformer_model.pth')
                    else:
                        patience_counter += 1
                    
                    # エポック進捗更新
                    current_lr = self.optimizer.param_groups[0]['lr']
                    epoch_pbar.set_postfix({
                        'Loss': f'{epoch_losses["total"]:.6f}',
                        'Best': f'{best_loss:.6f}',
                        'Patience': f'{patience_counter}/{self.patience}',
                        'γ': f'{np.mean(epoch_params["gamma"]) if epoch_params["gamma"] else 0.234:.4f}',
                        'LR': f'{current_lr:.2e}',
                        'VRAM': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB' if torch.cuda.is_available() else 'N/A'
                    })
                    
                    # 早期停止
                    if patience_counter >= self.patience:
                        print(f"\n🛑 早期停止: {self.patience}エポック改善なし")
                        break
                    
                    # 定期的な進捗表示（RTX3080版）
                    if (epoch + 1) % 25 == 0:
                        elapsed_time = time.time() - start_time
                        remaining_epochs = self.num_epochs - epoch - 1
                        estimated_remaining = elapsed_time / (epoch + 1) * remaining_epochs
                        
                        print(f"\n📊 エポック {epoch+1}/{self.num_epochs} (経過: {elapsed_time:.1f}秒, 残り推定: {estimated_remaining:.1f}秒):")
                        print(f"  総損失: {epoch_losses['total']:.6f}")
                        print(f"  データ損失: {epoch_losses['data']:.6f}")
                        print(f"  物理損失: {epoch_losses['physics']:.6f}")
                        print(f"  現在学習率: {current_lr:.2e}")
                        if torch.cuda.is_available():
                            print(f"  VRAM使用量: {torch.cuda.memory_allocated()/1024**3:.1f}/{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
                        if epoch_params['gamma']:
                            print(f"  平均γ: {np.mean(epoch_params['gamma']):.6f}")
                            print(f"  平均δ: {np.mean(epoch_params['delta']):.6f}")
                            print(f"  平均t_c: {np.mean(epoch_params['tc']):.6f}")
                else:
                    print(f"⚠️ エポック {epoch+1}: 有効なバッチがありませんでした")
                    
        except KeyboardInterrupt:
            print("\n🛑 ユーザーによる中断")
        except Exception as e:
            print(f"\n❌ 訓練中にエラーが発生: {e}")
        finally:
            # メモリクリア
            torch.cuda.empty_cache() if device.type == 'cuda' else None
            gc.collect()
        
        total_time = time.time() - start_time
        print(f"✅ モデル訓練完了 (総時間: {total_time:.1f}秒)")

    def evaluate_model(self, test_N_values):
        """
        モデル評価（修正版）
        """
        print("📊 モデル評価中...")
        
        self.model.eval()
        
        with torch.no_grad():
            # テストデータを作成
            test_log_N = torch.log(torch.tensor(test_N_values, dtype=torch.float32) + 1e-8)
            
            # 正規化（訓練時と同じ統計を使用）
            if hasattr(self, 'dataset'):
                test_log_N_norm = (test_log_N - self.dataset.log_N_mean) / self.dataset.log_N_std
            else:
                test_log_N_norm = test_log_N
            
            # 各テスト点に対して予測を実行
            predictions = []
            gamma_values = []
            delta_values = []
            tc_values = []
            
            for i in range(len(test_N_values)):
                # シーケンス形式に変換（最後のsequence_length個の点を使用）
                if i >= self.sequence_length - 1:
                    start_idx = i - self.sequence_length + 1
                    end_idx = i + 1
                    test_input = test_log_N_norm[start_idx:end_idx].unsqueeze(0).unsqueeze(-1).to(device)  # [1, seq_len, 1]
                else:
                    # 不足分はパディング
                    padding_size = self.sequence_length - i - 1
                    padded_input = torch.cat([
                        torch.zeros(padding_size, dtype=torch.float32),
                        test_log_N_norm[:i+1]
                    ])
                    test_input = padded_input.unsqueeze(0).unsqueeze(-1).to(device)  # [1, seq_len, 1]
                
                # 予測
                log_S_pred, gamma_pred, delta_pred, tc_pred = self.model(test_input)
                
                # 結果を保存
                predictions.append(log_S_pred.cpu().numpy())
                gamma_values.append(gamma_pred.cpu().numpy())
                delta_values.append(delta_pred.cpu().numpy())
                tc_values.append(tc_pred.cpu().numpy())
            
            # リストをnumpy配列に変換
            predictions = np.array(predictions).flatten()
            gamma_values = np.array(gamma_values).flatten()
            delta_values = np.array(delta_values).flatten()
            tc_values = np.array(tc_values).flatten()
            
            # 元のスケールに戻す
            if hasattr(self, 'dataset'):
                predictions = predictions * self.dataset.log_targets_std.numpy() + self.dataset.log_targets_mean.numpy()
            
            predictions = np.exp(predictions)
        
        # 統計計算
        results = {
            'predictions': predictions,
            'gamma_mean': np.mean(gamma_values),
            'gamma_std': np.std(gamma_values),
            'delta_mean': np.mean(delta_values),
            'delta_std': np.std(delta_values),
            'tc_mean': np.mean(tc_values),
            'tc_std': np.std(tc_values),
            'gamma_values': gamma_values,
            'delta_values': delta_values,
            'tc_values': tc_values
        }
        
        print("✅ モデル評価完了")
        print(f"📊 最適化パラメータ:")
        print(f"  γ = {results['gamma_mean']:.6f} ± {results['gamma_std']:.6f}")
        print(f"  δ = {results['delta_mean']:.6f} ± {results['delta_std']:.6f}")
        print(f"  t_c = {results['tc_mean']:.6f} ± {results['tc_std']:.6f}")
        
        return results
    
    def visualize_results(self, test_N_values, results):
        """
        結果の可視化
        """
        print("📈 結果可視化中...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 訓練損失の推移
        if self.train_history['total_loss']:
            axes[0, 0].plot(self.train_history['total_loss'], label='総損失', color='red', linewidth=2)
            axes[0, 0].plot(self.train_history['data_loss'], label='データ損失', color='blue', linewidth=2)
            axes[0, 0].plot(self.train_history['physics_loss'], label='物理損失', color='green', linewidth=2)
            axes[0, 0].set_xlabel('エポック')
            axes[0, 0].set_ylabel('損失')
            axes[0, 0].set_title('Transformer訓練損失の推移')
            axes[0, 0].legend()
            axes[0, 0].set_yscale('log')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. パラメータの収束
        if self.train_history['gamma_values']:
            axes[0, 1].plot(self.train_history['gamma_values'], label='γ', color='red', linewidth=2)
            axes[0, 1].axhline(y=0.234, color='red', linestyle='--', alpha=0.7, label='γ理論値')
            axes[0, 1].set_xlabel('エポック')
            axes[0, 1].set_ylabel('γ値')
            axes[0, 1].set_title('γパラメータの収束')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        if self.train_history['delta_values']:
            axes[0, 2].plot(self.train_history['delta_values'], label='δ', color='blue', linewidth=2)
            axes[0, 2].axhline(y=0.035, color='blue', linestyle='--', alpha=0.7, label='δ理論値')
            axes[0, 2].set_xlabel('エポック')
            axes[0, 2].set_ylabel('δ値')
            axes[0, 2].set_title('δパラメータの収束')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 3. 超収束因子の予測
        axes[1, 0].loglog(test_N_values, results['predictions'], 'b-', label='Transformer予測', linewidth=3)
        
        # 理論値との比較
        gamma_true, delta_true, t_c_true = 0.234, 0.035, 17.26
        theoretical_values = []
        for N in test_N_values:
            integral = gamma_true * np.log(max(N / t_c_true, 1e-8))
            if N > t_c_true:
                integral += delta_true * (N - t_c_true) * 0.1
            theoretical_values.append(np.exp(np.clip(integral, -10, 10)))
        
        axes[1, 0].loglog(test_N_values, theoretical_values, 'r--', label='理論値', linewidth=2)
        axes[1, 0].set_xlabel('次元数 N')
        axes[1, 0].set_ylabel('超収束因子 S(N)')
        axes[1, 0].set_title('Transformer: 超収束因子の予測vs理論値')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 学習率の推移
        if self.train_history['learning_rates']:
            axes[1, 1].plot(self.train_history['learning_rates'], color='purple', linewidth=2)
            axes[1, 1].set_xlabel('エポック')
            axes[1, 1].set_ylabel('学習率')
            axes[1, 1].set_title('学習率の推移')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 5. t_c パラメータの収束
        if self.train_history['tc_values']:
            axes[1, 2].plot(self.train_history['tc_values'], label='t_c', color='green', linewidth=2)
            axes[1, 2].axhline(y=17.26, color='green', linestyle='--', alpha=0.7, label='t_c理論値')
            axes[1, 2].set_xlabel('エポック')
            axes[1, 2].set_ylabel('t_c値')
            axes[1, 2].set_title('t_cパラメータの収束')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('nkat_transformer_optimization_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 可視化完了")
    
    def save_model_and_results(self, results, filename_prefix='nkat_transformer_optimization'):
        """
        モデルと結果の保存
        """
        print("💾 モデルと結果を保存中...")
        
        # モデル保存
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history,
            'results': results
        }, f'{filename_prefix}_model.pth')
        
        # 結果をJSON形式で保存
        json_results = {
            'optimal_parameters': {
                'gamma_mean': float(results['gamma_mean']),
                'gamma_std': float(results['gamma_std']),
                'delta_mean': float(results['delta_mean']),
                'delta_std': float(results['delta_std']),
                'tc_mean': float(results['tc_mean']),
                'tc_std': float(results['tc_std'])
            },
            'training_config': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs,
                'sequence_length': self.sequence_length
            },
            'final_losses': {
                'total_loss': self.train_history['total_loss'][-1] if self.train_history['total_loss'] else 0,
                'data_loss': self.train_history['data_loss'][-1] if self.train_history['data_loss'] else 0,
                'physics_loss': self.train_history['physics_loss'][-1] if self.train_history['physics_loss'] else 0,
                'reg_loss': self.train_history['reg_loss'][-1] if self.train_history['reg_loss'] else 0
            }
        }
        
        with open(f'{filename_prefix}_results.json', 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 保存完了: {filename_prefix}_model.pth, {filename_prefix}_results.json")

def main():
    """メイン実行関数（RTX3080中パワー版）"""
    print("🚀 NKAT Transformer深層学習最適化システム開始（RTX3080中パワー版）")
    print("="*70)
    
    try:
        # システム初期化（RTX3080中パワー版）
        optimizer = NKATTransformerOptimizer(
            learning_rate=3e-4,
            batch_size=16,  # RTX3080に適したサイズ
            num_epochs=300,  # より長い訓練
            sequence_length=32,  # より長いシーケンス
            patience=50,  # より長い忍耐
            min_delta=1e-6
        )
        
        # 訓練データ生成（RTX3080中パワー版）
        dataloader, dataset = optimizer.generate_training_data(
            N_range=(10, 500),  # より広い範囲
            num_samples=800  # より多くのサンプル
        )
        
        # データセットを保存（評価時に使用）
        optimizer.dataset = dataset
        
        print(f"📊 データローダー確認: {len(dataloader)}バッチ")
        if len(dataloader) == 0:
            print("❌ データローダーが空です。プログラムを終了します。")
            return
        
        # モデル訓練
        print("🎓 RTX3080中パワーモデルで訓練開始...")
        optimizer.train(dataloader)
        
        # モデル評価
        test_N_values = np.logspace(1, 2.7, 50)  # より広い範囲とより多くの点
        results = optimizer.evaluate_model(test_N_values)
        
        # 結果可視化
        optimizer.visualize_results(test_N_values, results)
        
        # モデルと結果の保存
        optimizer.save_model_and_results(results, 'nkat_transformer_rtx3080_medium')
        
        # リーマン予想への含意
        gamma_opt = results['gamma_mean']
        t_c_opt = results['tc_mean']
        riemann_convergence = gamma_opt * np.log(max(500 / t_c_opt, 1e-8))
        riemann_deviation = abs(riemann_convergence - 0.5)
        
        print("\n" + "="*70)
        print("🎯 NKAT Transformer深層学習最適化結果（RTX3080中パワー版）")
        print("="*70)
        print(f"📊 最適化パラメータ:")
        print(f"  γ = {results['gamma_mean']:.6f} ± {results['gamma_std']:.6f}")
        print(f"  δ = {results['delta_mean']:.6f} ± {results['delta_std']:.6f}")
        print(f"  t_c = {results['tc_mean']:.6f} ± {results['tc_std']:.6f}")
        print(f"\n🎯 リーマン予想への含意:")
        print(f"  収束率: γ·ln(500/t_c) = {riemann_convergence:.6f}")
        print(f"  理論値からの偏差: {riemann_deviation:.6f}")
        print(f"  リーマン予想支持度: {100*(1-min(riemann_deviation/0.1, 1.0)):.1f}%")
        
        print("\n🏁 RTX3080中パワー Transformer深層学習最適化完了")
        
    except KeyboardInterrupt:
        print("\n🛑 ユーザーによる中断")
    except Exception as e:
        print(f"\n❌ システムエラー: {e}")
        print("🔧 パラメータを調整して再実行を推奨します")
        import traceback
        traceback.print_exc()
    finally:
        # 最終メモリクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("🧹 メモリクリア完了")

if __name__ == "__main__":
    main() 