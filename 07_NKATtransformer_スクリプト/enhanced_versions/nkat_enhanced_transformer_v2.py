#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced NKAT-Transformer v2.0 - 99%+ Accuracy Target
改善版NKAT Vision Transformer - 97.79% → 99%+ 精度向上

主要改善点:
1. 深層化されたモデル構造（512次元、12層）
2. 混合精度問題の解決
3. 困難クラス(5,7,9)専用対策
4. 強化データ拡張パイプライン
5. 安定した訓練戦略

Author: NKAT Advanced Computing Team
Date: 2025-06-01
GPU: RTX3080 CUDA Optimized
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
from tqdm import tqdm
import logging
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# 英語表記設定（文字化け防止）
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/nkat_enhanced_v2_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedNKATConfig:
    """Enhanced NKAT-Transformer Configuration"""
    
    def __init__(self):
        # 基本設定
        self.image_size = 28
        self.patch_size = 7
        self.num_patches = (self.image_size // self.patch_size) ** 2  # 16
        self.channels = 1
        self.num_classes = 10
        
        # 改善版モデル設定
        self.d_model = 512      # 384 → 512 (強化)
        self.nhead = 8          # 6 → 8 (強化)
        self.num_layers = 12    # 8 → 12 (深層化)
        self.dim_feedforward = 2048  # 1536 → 2048 (強化)
        self.dropout = 0.08     # 0.1 → 0.08 (最適化)
        
        # 安定性設定
        self.use_mixed_precision = False  # 安定性のため無効化
        self.use_gradient_clipping = True
        self.max_grad_norm = 1.0
        
        # 困難クラス対策
        self.use_class_weights = True
        self.difficult_classes = [5, 7, 9]
        self.class_weight_boost = 1.5
        
        # 強化データ拡張
        self.use_advanced_augmentation = True
        self.rotation_range = 15
        self.zoom_range = 0.15
        self.use_elastic_deformation = True
        self.use_random_erasing = True
        
        # 訓練戦略
        self.num_epochs = 100
        self.batch_size = 64    # 128 → 64 (安定性)
        self.learning_rate = 1e-4  # 3e-4 → 1e-4 (安定性)
        self.use_cosine_restart = True
        self.restart_period = 20
        
        # 正則化
        self.use_label_smoothing = True
        self.label_smoothing = 0.08
        self.use_mixup = True
        self.mixup_alpha = 0.4
        self.weight_decay = 2e-4
        
        # アンサンブル
        self.save_multiple_models = True
        self.model_variants = 3

class AdvancedDataAugmentation:
    """強化版データ拡張パイプライン"""
    
    def __init__(self, config):
        self.config = config
        
    def create_train_transforms(self):
        """訓練用変換（強化版）"""
        transforms_list = [
            # MNISTはすでにPILImageなのでToPILImage()は不要
            
            # 強化された回転
            transforms.RandomRotation(
                degrees=self.config.rotation_range,
                fill=0,
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            
            # アフィン変換（微調整）
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5,
                fill=0
            ),
            
            # 透視変換（軽微）
            transforms.RandomPerspective(
                distortion_scale=0.1,
                p=0.3,
                fill=0
            ),
            
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
        
        # ランダム消去
        if self.config.use_random_erasing:
            transforms_list.append(
                transforms.RandomErasing(
                    p=0.1,
                    scale=(0.02, 0.1),
                    ratio=(0.3, 3.3),
                    value=0
                )
            )
        
        return transforms.Compose(transforms_list)
    
    def create_test_transforms(self):
        """テスト用変換"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

class EnhancedPatchEmbedding(nn.Module):
    """改善版パッチ埋め込み"""
    
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.d_model = config.d_model
        
        # 段階的特徴抽出
        self.conv_layers = nn.Sequential(
            # 第1段階：低レベル特徴
            nn.Conv2d(config.channels, config.d_model // 4, 3, padding=1),
            nn.BatchNorm2d(config.d_model // 4),
            nn.GELU(),
            
            # 第2段階：中レベル特徴
            nn.Conv2d(config.d_model // 4, config.d_model // 2, 3, padding=1),
            nn.BatchNorm2d(config.d_model // 2),
            nn.GELU(),
            
            # 第3段階：高レベル特徴 + パッチ化
            nn.Conv2d(config.d_model // 2, config.d_model, 
                     config.patch_size, stride=config.patch_size)
        )
        
    def forward(self, x):
        # x: (B, C, H, W) -> (B, d_model, H//patch_size, W//patch_size)
        x = self.conv_layers(x)
        # Flatten spatial dimensions: (B, d_model, num_patches)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).transpose(1, 2)  # (B, num_patches, d_model)
        return x

class EnhancedNKATVisionTransformer(nn.Module):
    """Enhanced NKAT Vision Transformer v2.0"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 改善版パッチ埋め込み
        self.patch_embedding = EnhancedPatchEmbedding(config)
        
        # 位置埋め込み（学習可能）
        self.pos_embedding = nn.Parameter(
            torch.randn(1, config.num_patches + 1, config.d_model) * 0.02
        )
        
        # クラストークン
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, config.d_model) * 0.02
        )
        
        # 入力正規化
        self.input_norm = nn.LayerNorm(config.d_model)
        
        # Enhanced Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        
        # 深層分類ヘッド
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout * 0.5),
            
            # 第1層
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            
            # 第2層
            nn.Linear(config.d_model // 2, config.d_model // 4),
            nn.GELU(),
            nn.Dropout(config.dropout * 0.5),
            
            # 出力層
            nn.Linear(config.d_model // 4, config.num_classes)
        )
        
        # 重み初期化
        self.apply(self._init_weights)
        
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
    
    def forward(self, x):
        B = x.shape[0]
        
        # パッチ埋め込み
        x = self.patch_embedding(x)  # (B, num_patches, d_model)
        
        # クラストークン追加
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, d_model)
        
        # 位置埋め込み追加
        x = x + self.pos_embedding
        x = self.input_norm(x)
        
        # Transformer処理
        x = self.transformer(x)
        
        # クラストークンを抽出して分類
        cls_output = x[:, 0]  # (B, d_model)
        logits = self.classifier(cls_output)
        
        return logits

def create_class_weighted_sampler(dataset, config):
    """困難クラス対策用重み付きサンプラー"""
    # 簡単なアプローチ：変換なしでラベルを取得
    targets = torch.tensor([dataset.targets[i] for i in range(len(dataset))])
    
    # クラス別サンプル数カウント
    class_counts = torch.zeros(config.num_classes)
    for label in targets:
        class_counts[label] += 1
    
    # 基本重みの計算（逆頻度）
    class_weights = 1.0 / class_counts
    
    # 困難クラスの重みを強化
    for difficult_class in config.difficult_classes:
        class_weights[difficult_class] *= config.class_weight_boost
    
    # サンプル重み
    sample_weights = [class_weights[label] for label in targets]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

def mixup_data(x, y, alpha=1.0):
    """Mixup データ拡張"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup用損失関数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class EnhancedTrainer:
    """Enhanced Training Pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # モデル作成
        self.model = EnhancedNKATVisionTransformer(config).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Enhanced model parameters: {total_params:,}")
        
        # データローダー作成
        self.train_loader, self.test_loader = self._create_dataloaders()
        
        # 最適化設定
        self.optimizer, self.scheduler = self._create_optimizer()
        self.criterion = self._create_criterion()
        
        # 訓練記録
        self.train_history = {
            'loss': [], 'accuracy': [], 'lr': []
        }
        self.test_history = {
            'loss': [], 'accuracy': []
        }
        
    def _create_dataloaders(self):
        """データローダー作成"""
        augmentation = AdvancedDataAugmentation(self.config)
        
        # 訓練データ
        train_transform = augmentation.create_train_transforms()
        train_dataset = torchvision.datasets.MNIST(
            root="data", train=True, download=True, transform=train_transform
        )
        
        if self.config.use_class_weights:
            sampler = create_class_weighted_sampler(train_dataset, self.config)
            shuffle = False
        else:
            sampler = None
            shuffle = True
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        # テストデータ
        test_transform = augmentation.create_test_transforms()
        test_dataset = torchvision.datasets.MNIST(
            root="data", train=False, download=True, transform=test_transform
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
        return train_loader, test_loader
    
    def _create_optimizer(self):
        """最適化器作成"""
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        if self.config.use_cosine_restart:
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=self.config.restart_period,
                T_mult=2,
                eta_min=self.config.learning_rate * 0.01
            )
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.learning_rate * 0.1
            )
        
        return optimizer, scheduler
    
    def _create_criterion(self):
        """損失関数作成"""
        if self.config.use_label_smoothing:
            criterion = nn.CrossEntropyLoss(
                label_smoothing=self.config.label_smoothing
            )
        else:
            criterion = nn.CrossEntropyLoss()
        
        return criterion
    
    def train_epoch(self, epoch):
        """1エポック訓練"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Mixup適用
            if self.config.use_mixup and np.random.random() < 0.5:
                data, target_a, target_b, lam = mixup_data(data, target, self.config.mixup_alpha)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = mixup_criterion(self.criterion, output, target_a, target_b, lam)
            else:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
            
            loss.backward()
            
            # 勾配クリッピング
            if self.config.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
            
            self.optimizer.step()
            
            # 統計更新
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # 進捗表示
            if batch_idx % 100 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%',
                    'LR': f'{current_lr:.6f}'
                })
        
        # エポック統計
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        current_lr = self.optimizer.param_groups[0]['lr']
        
        self.train_history['loss'].append(avg_loss)
        self.train_history['accuracy'].append(accuracy)
        self.train_history['lr'].append(current_lr)
        
        return avg_loss, accuracy
    
    def evaluate(self):
        """テスト評価"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc='Evaluating'):
                data, target = data.to(self.device), target.to(self.device)
                
                # 混合精度を使用せず安定評価
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        
        self.test_history['loss'].append(avg_loss)
        self.test_history['accuracy'].append(accuracy)
        
        return avg_loss, accuracy, all_preds, all_targets
    
    def train(self):
        """完全訓練パイプライン"""
        logger.info("Starting Enhanced NKAT-Transformer Training")
        logger.info("Target: 99%+ Test Accuracy")
        
        best_accuracy = 0
        patience_counter = 0
        patience = 15
        
        for epoch in range(self.config.num_epochs):
            # 訓練
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 評価
            test_loss, test_acc, preds, targets = self.evaluate()
            
            # スケジューラー更新
            self.scheduler.step()
            
            # ログ出力
            logger.info(f'Epoch {epoch+1:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                       f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
            
            # ベストモデル保存
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'test_accuracy': test_acc,
                    'config': self.config.__dict__
                }
                
                torch.save(checkpoint, 'checkpoints/nkat_enhanced_v2_best.pth')
                logger.info(f'New best model saved! Accuracy: {test_acc:.2f}%')
                
                # 詳細分析（ベスト時のみ）
                if test_acc > 98.5:  # 98.5%を超えた場合のみ詳細分析
                    self.detailed_analysis(preds, targets, epoch)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f'Early stopping at epoch {epoch+1}')
                break
            
            # 目標達成チェック
            if test_acc >= 99.0:
                logger.info(f'🎉 TARGET ACHIEVED! 99%+ Accuracy: {test_acc:.2f}%')
                break
        
        logger.info(f'Training completed! Best accuracy: {best_accuracy:.2f}%')
        return best_accuracy
    
    def detailed_analysis(self, preds, targets, epoch):
        """詳細分析と可視化"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 混同行列
        cm = confusion_matrix(targets, preds)
        
        # クラス別精度計算
        class_accuracies = {}
        for i in range(10):
            class_mask = np.array(targets) == i
            if np.sum(class_mask) > 0:
                class_acc = np.sum(np.array(preds)[class_mask] == i) / np.sum(class_mask) * 100
                class_accuracies[i] = class_acc
        
        # 可視化
        plt.figure(figsize=(15, 10))
        
        # 混同行列
        plt.subplot(2, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(10), yticklabels=range(10))
        plt.title(f'Confusion Matrix - Epoch {epoch+1}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # クラス別精度
        plt.subplot(2, 2, 2)
        classes = list(class_accuracies.keys())
        accuracies = list(class_accuracies.values())
        colors = ['red' if cls in self.config.difficult_classes else 'blue' for cls in classes]
        
        bars = plt.bar(classes, accuracies, color=colors, alpha=0.7)
        plt.title('Class-wise Accuracy')
        plt.xlabel('Class')
        plt.ylabel('Accuracy (%)')
        plt.ylim(95, 100)
        
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        # 訓練履歴
        plt.subplot(2, 2, 3)
        plt.plot(self.train_history['accuracy'], label='Train', color='blue')
        plt.plot(self.test_history['accuracy'], label='Test', color='red')
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 学習率履歴
        plt.subplot(2, 2, 4)
        plt.plot(self.train_history['lr'], color='green')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'figures/nkat_enhanced_v2_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # レポート保存
        report = {
            'timestamp': timestamp,
            'epoch': epoch + 1,
            'test_accuracy': float(np.mean(np.array(preds) == np.array(targets)) * 100),
            'class_accuracies': {str(k): float(v) for k, v in class_accuracies.items()},
            'confusion_matrix': cm.tolist(),
            'difficult_classes_performance': {
                str(cls): float(class_accuracies.get(cls, 0)) 
                for cls in self.config.difficult_classes
            }
        }
        
        with open(f'analysis/nkat_enhanced_v2_report_{timestamp}.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f'Detailed analysis saved: {timestamp}')

def main():
    """メイン実行関数"""
    print("🚀 Enhanced NKAT-Transformer v2.0 Training")
    print("=" * 60)
    print("Target: 97.79% → 99%+ Accuracy")
    print("GPU: RTX3080 CUDA Optimized")
    print("=" * 60)
    
    # ディレクトリ作成
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    os.makedirs('analysis', exist_ok=True)
    
    # 設定
    config = EnhancedNKATConfig()
    
    print(f"\n📊 Enhanced Model Configuration:")
    print(f"• d_model: {config.d_model} (384→512)")
    print(f"• Layers: {config.num_layers} (8→12)")
    print(f"• Attention heads: {config.nhead} (6→8)")
    print(f"• Mixed precision: {config.use_mixed_precision} (Disabled for stability)")
    print(f"• Class weights: {config.use_class_weights} (Difficult classes: {config.difficult_classes})")
    print(f"• Advanced augmentation: {config.use_advanced_augmentation}")
    print(f"• Training epochs: {config.num_epochs}")
    
    # CUDA情報
    if torch.cuda.is_available():
        print(f"\n🎮 GPU Information:")
        print(f"• Device: {torch.cuda.get_device_name(0)}")
        print(f"• Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"• CUDA Version: {torch.version.cuda}")
    
    # 訓練開始
    trainer = EnhancedTrainer(config)
    best_accuracy = trainer.train()
    
    print(f"\n🎯 Training Results:")
    print(f"• Best Test Accuracy: {best_accuracy:.2f}%")
    print(f"• Target Achievement: {'✅ SUCCESS' if best_accuracy >= 99.0 else '🔄 CONTINUE'}")
    
    if best_accuracy >= 99.0:
        print("\n🎉 CONGRATULATIONS!")
        print("Enhanced NKAT-Transformer has achieved 99%+ accuracy!")
        print("Ready for production deployment.")
    else:
        print(f"\n📈 Progress: {best_accuracy:.2f}% / 99.0%")
        print("Consider further optimization strategies.")
    
    print("\n" + "=" * 60)
    print("Enhanced NKAT-Transformer v2.0 Training Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main() 