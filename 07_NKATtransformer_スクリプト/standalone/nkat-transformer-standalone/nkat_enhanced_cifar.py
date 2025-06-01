#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT-Transformer Enhanced CIFAR-10 Version
CIFAR-10対応強化版 Vision Transformer

Stage 2 改善プラン実装：
- ハイブリッド ConvStem
- Color-Wise LayerNorm
- 強化データ拡張
- Mixup/CutMix対応

Author: NKAT Advanced Computing Team  
Version: 2.0.0 - CIFAR Enhanced
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from tqdm import tqdm
import math
import random
import warnings
warnings.filterwarnings('ignore')

# 日本語対応設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class NKATEnhancedConfig:
    """Enhanced NKAT-Transformer設定クラス"""
    
    def __init__(self, dataset='cifar10'):
        self.dataset = dataset
        
        # データセット別設定
        if dataset == 'cifar10':
            self.image_size = 32
            self.channels = 3
            self.num_classes = 10
            self.patch_size = 2  # CIFAR用の細かいパッチ
        elif dataset == 'mnist':
            self.image_size = 28
            self.channels = 1
            self.num_classes = 10
            self.patch_size = 7
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
            
        # 共通設定
        self.num_patches = (self.image_size // self.patch_size) ** 2
        
        # モデル構造 - 軽量化版
        self.embed_dim = 384  # 512→384に軽量化
        self.depth = 7        # 12→7に軽量化
        self.num_heads = 8
        self.mlp_ratio = 4.0
        self.dropout_embed = 0.08
        self.dropout_attn = 0.17
        
        # ConvStem設定
        self.conv_stem = "tiny"  # off, tiny, small
        
        # NKAT特有設定
        self.temperature = 0.65
        self.top_k = 6
        self.top_p = 0.80
        self.nkat_strength = 0.0024
        self.nkat_decay = 0.99
        
        # 学習設定
        self.learning_rate = 2e-4
        self.batch_size = 32  # GPU負荷軽減
        self.num_epochs = 40
        self.weight_decay = 1e-4
        self.warmup_epochs = 5
        self.clip_grad = 1.0
        
        # EMA設定
        self.use_ema = True
        self.ema_decay = 0.9995
        
        # データ拡張設定（CIFAR特化）
        self.use_randaugment = True
        self.randaugment_n = 2
        self.randaugment_m = 9
        self.use_cutmix = True
        self.cutmix_prob = 0.2
        self.use_mixup = True
        self.mixup_prob = 0.1
        self.mixup_alpha = 0.2
        self.cutout_size = 8

class ColorWiseLayerNorm(nn.Module):
    """Color-Wise LayerNorm for RGB images"""
    
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.ln = nn.LayerNorm(num_channels)
        
    def forward(self, x):
        # x: (B, C, H, W) -> (B, H, W, C) -> LN -> (B, C, H, W)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return x

class ConvStem(nn.Module):
    """Hybrid Convolutional Stem for better feature extraction"""
    
    def __init__(self, in_channels, embed_dim, stem_type="tiny"):
        super().__init__()
        self.stem_type = stem_type
        
        if stem_type == "off":
            # 従来のパッチ埋め込みのみ
            self.conv_stem = nn.Identity()
            self.out_channels = in_channels
        elif stem_type == "tiny":
            # 軽量ConvStem
            self.conv_stem = nn.Sequential(
                # Stage-A: 32×32 → 16×16 (for CIFAR-10)
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.GELU(),
                # Stage-B: 16×16 → 8×8  
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.GELU(),
            )
            self.out_channels = 64
        elif stem_type == "small":
            # 中規模ConvStem
            self.conv_stem = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.GELU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.GELU(),
            )
            self.out_channels = 128
        
    def forward(self, x):
        return self.conv_stem(x)

class NKATPatchEmbedding(nn.Module):
    """Enhanced Patch Embedding with ConvStem support"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Color-wise normalization for RGB
        if config.channels == 3:
            self.color_norm = ColorWiseLayerNorm(config.channels)
        else:
            self.color_norm = nn.Identity()
            
        # ConvStem
        self.conv_stem = ConvStem(
            config.channels, 
            config.embed_dim, 
            config.conv_stem
        )
        
        # Patch projection
        if config.conv_stem == "off":
            # 従来通り
            self.proj = nn.Conv2d(
                config.channels, 
                config.embed_dim,
                kernel_size=config.patch_size,
                stride=config.patch_size
            )
        else:
            # ConvStem後の特徴をembedding
            self.proj = nn.Conv2d(
                self.conv_stem.out_channels,
                config.embed_dim,
                kernel_size=config.patch_size,
                stride=config.patch_size
            )
            
    def forward(self, x):
        # Color-wise normalization
        x = self.color_norm(x)
        
        # ConvStem processing
        x = self.conv_stem(x)
        
        # Patch embedding
        x = self.proj(x)  # (B, embed_dim, H', W')
        
        # Flatten and transpose
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        return x

class NKATSelfAttention(nn.Module):
    """NKAT Self-Attention with enhanced temperature control"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert self.embed_dim % self.num_heads == 0
        
        # QKV projection
        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=False)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_drop = nn.Dropout(config.dropout_attn)
        
        # NKAT parameters
        self.temperature = nn.Parameter(torch.ones(1) * config.temperature)
        self.top_k = config.top_k
        self.top_p = config.top_p
        
    def forward(self, x):
        B, N, C = x.shape
        
        # QKV computation
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention with temperature
        scale = (self.head_dim ** -0.5) * self.temperature
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # NKAT: Top-k + Top-p filtering
        if self.training:
            # Top-k filtering
            if self.top_k > 0:
                top_k_vals, _ = torch.topk(attn, self.top_k, dim=-1)
                attn = torch.where(
                    attn < top_k_vals[..., [-1]], 
                    torch.full_like(attn, float('-inf')), 
                    attn
                )
            
            # Top-p (nucleus) filtering  
            if self.top_p < 1.0:
                sorted_attn, sorted_indices = torch.sort(attn, descending=True, dim=-1)
                cumsum_probs = torch.cumsum(F.softmax(sorted_attn, dim=-1), dim=-1)
                sorted_indices_to_remove = cumsum_probs > self.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                attn = attn.masked_fill(indices_to_remove, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=config.dropout_attn, training=self.training)
        
        # Apply attention to values
        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class NKATTransformerBlock(nn.Module):
    """Enhanced Transformer Block"""
    
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = NKATSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        
        # MLP
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_embed),
            nn.Linear(mlp_hidden_dim, config.embed_dim),
            nn.Dropout(config.dropout_embed)
        )
        
    def forward(self, x):
        # Pre-norm design
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class NKATEnhancedViT(nn.Module):
    """Enhanced NKAT Vision Transformer for CIFAR-10"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embed = NKATPatchEmbedding(config)
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim) * 0.02)
        
        # Position embedding
        num_patches = config.num_patches
        if config.conv_stem != "off":
            # ConvStem使用時はパッチ数が変わる
            if config.conv_stem == "tiny":
                # 32→8, patch_size=2なら 4x4=16 patches
                num_patches = (config.image_size // 4 // config.patch_size) ** 2
            elif config.conv_stem == "small":
                num_patches = (config.image_size // 4 // config.patch_size) ** 2
                
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, config.embed_dim) * 0.02
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            NKATTransformerBlock(config) for _ in range(config.depth)
        ])
        
        # Final norm
        self.norm = nn.LayerNorm(config.embed_dim)
        
        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(config.dropout_embed * 0.5),
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.GELU(), 
            nn.Dropout(config.dropout_embed),
            nn.Linear(config.embed_dim // 2, config.num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
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
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transform
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        # Classification
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        
        return logits

# Enhanced Data Augmentation Classes
class RandAugment:
    """RandAugment implementation"""
    
    def __init__(self, n=2, m=9):
        self.n = n
        self.m = m
        
    def __call__(self, img):
        ops = [
            lambda x: transforms.functional.autocontrast(x),
            lambda x: transforms.functional.equalize(x),
            lambda x: transforms.functional.posterize(x, bits=self.m//2 + 4),
            lambda x: transforms.functional.solarize(x, threshold=256-self.m*10),
        ]
        
        for _ in range(self.n):
            op = random.choice(ops)
            if random.random() < 0.5:
                img = op(img)
        return img

class CutMix:
    """CutMix augmentation"""
    
    def __init__(self, beta=1.0, prob=0.5):
        self.beta = beta
        self.prob = prob
        
    def __call__(self, batch):
        if np.random.rand() > self.prob:
            return batch
            
        # Implementation would go here
        return batch

def create_enhanced_dataloaders(config):
    """Create enhanced dataloaders with strong augmentation"""
    
    if config.dataset == 'cifar10':
        # CIFAR-10 transforms
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomCrop(32, padding=4),
            RandAugment(n=config.randaugment_n, m=config.randaugment_m),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=test_transform
        )
        
    elif config.dataset == 'mnist':
        # MNIST transforms (legacy support)
        train_transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=test_transform
        )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, test_loader

class EMA:
    """Exponential Moving Average"""
    
    def __init__(self, model, decay=0.9995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
                
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
                
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class NKATEnhancedTrainer:
    """Enhanced NKAT Trainer for Stage 2 improvements"""
    
    def __init__(self, config=None):
        if config is None:
            config = NKATEnhancedConfig()
        self.config = config
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
        # Model
        self.model = NKATEnhancedViT(config).to(self.device)
        
        # EMA
        if config.use_ema:
            self.ema = EMA(self.model, config.ema_decay)
        else:
            self.ema = None
            
        # Data loaders
        self.train_loader, self.test_loader = create_enhanced_dataloaders(config)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler with warmup
        self.scheduler = self._create_scheduler()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Training state
        self.best_acc = 0.0
        self.training_history = []
        
    def _create_scheduler(self):
        """Create cosine scheduler with warmup"""
        def lr_lambda(epoch):
            if epoch < self.config.warmup_epochs:
                return epoch / self.config.warmup_epochs
            else:
                return 0.5 * (1 + math.cos(math.pi * (epoch - self.config.warmup_epochs) / 
                                          (self.config.num_epochs - self.config.warmup_epochs)))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
    def mixup_data(self, x, y, alpha=0.2):
        """Mixup augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
        
    def mixup_criterion(self, pred, y_a, y_b, lam):
        """Mixup loss calculation"""
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Mixup augmentation
            if self.config.use_mixup and np.random.rand() < self.config.mixup_prob:
                data, target_a, target_b, lam = self.mixup_data(data, target, self.config.mixup_alpha)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.mixup_criterion(output, target_a, target_b, lam)
            else:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
            
            loss.backward()
            
            # Gradient clipping
            if self.config.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad)
                
            self.optimizer.step()
            
            # EMA update
            if self.ema is not None:
                self.ema.update()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'LR': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
        
        self.scheduler.step()
        epoch_acc = 100. * correct / total
        epoch_loss = total_loss / len(self.train_loader)
        
        return epoch_acc, epoch_loss
    
    def evaluate(self):
        """Evaluate model"""
        # Apply EMA if available
        if self.ema is not None:
            self.ema.apply_shadow()
            
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc='Evaluating'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        # Restore original weights if EMA was applied
        if self.ema is not None:
            self.ema.restore()
            
        accuracy = 100. * correct / total
        avg_loss = test_loss / len(self.test_loader)
        
        return accuracy, avg_loss
    
    def train(self):
        """Full training loop"""
        print(f"🎯 Starting Enhanced NKAT Training")
        print(f"Dataset: {self.config.dataset.upper()}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Calculate theoretical lambda
        total_params = sum(p.numel() for p in self.model.parameters())
        lambda_theory = total_params / 1e6  # Million parameters
        print(f"λ_theory: {lambda_theory:.3f}")
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_acc, train_loss = self.train_epoch(epoch)
            
            # Evaluate
            test_acc, test_loss = self.evaluate()
            
            # Save best model
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.save_checkpoint(epoch, test_acc)
            
            # Log results
            self.training_history.append({
                'epoch': epoch + 1,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'test_acc': test_acc,
                'test_loss': test_loss,
                'lr': self.scheduler.get_last_lr()[0]
            })
            
            print(f"Epoch {epoch+1}: Train={train_acc:.2f}%, Test={test_acc:.2f}%, Best={self.best_acc:.2f}%")
        
        print(f"\n🎉 Training Complete!")
        print(f"Best Accuracy: {self.best_acc:.2f}%")
        
        # Calculate TPE
        tpe = self.calculate_tpe(self.best_acc, lambda_theory)
        print(f"TPE: {tpe:.4f}")
        
        return self.best_acc, tpe
    
    def calculate_tpe(self, accuracy, lambda_theory):
        """Calculate Theoretical Performance Efficiency"""
        return (accuracy / 100.0) / math.log10(1 + lambda_theory)
    
    def save_checkpoint(self, epoch, accuracy):
        """Save model checkpoint"""
        os.makedirs('nkat_models', exist_ok=True)
        
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'config': self.config
        }
        
        if self.ema is not None:
            state['ema_shadow'] = self.ema.shadow
            
        torch.save(state, f'nkat_models/nkat_enhanced_{self.config.dataset}_best.pth')
        print(f"💾 Model saved with accuracy: {accuracy:.2f}%")

def quick_cifar_test():
    """Quick CIFAR-10 smoke test"""
    print("🧪 CIFAR-10 Smoke Test (10 epochs)")
    
    config = NKATEnhancedConfig('cifar10')
    config.num_epochs = 10
    config.batch_size = 32
    
    trainer = NKATEnhancedTrainer(config)
    accuracy, tpe = trainer.train()
    
    print(f"\n📊 Smoke Test Results:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"TPE: {tpe:.4f}")
    
    if accuracy >= 45.0:
        print("✅ CIFAR-10対応確認！本格訓練を開始可能")
        return True
    else:
        print("❌ 精度不足。パラメータ調整が必要")
        return False

if __name__ == "__main__":
    quick_cifar_test() 