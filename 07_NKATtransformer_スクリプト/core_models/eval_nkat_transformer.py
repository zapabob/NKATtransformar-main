#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT-Transformer最終評価スクリプト
RTX3080混合精度推論・高速化対応

Author: NKAT Advanced Computing Team
Date: 2025-01-26
CUDA Requirement: RTX3080 or higher
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import time
import os
from datetime import datetime

# NKAT実装のインポート
from nkat_transformer_mnist_recognition import NKATVisionTransformer, NKATVisionConfig

# CUDA最適化設定
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# 日本語文字化け防止・英語表記設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

def load_best_model():
    """最高性能のNKAT-ViTモデルを読み込み"""
    print("Loading best NKAT-ViT model...")
    
    config = NKATVisionConfig()
    model = NKATVisionTransformer(config).to(device)
    
    # チェックポイントの読み込み
    checkpoint_path = "nkat_mnist_checkpoints/latest_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        val_acc = checkpoint.get('val_acc', 'unknown')
        print(f"Loaded checkpoint from epoch {epoch}")
        if isinstance(val_acc, (int, float)):
            print(f"Best validation accuracy: {val_acc:.4f}%")
        else:
            print(f"Best validation accuracy: {val_acc}")
    else:
        # 最新のチェックポイントを探す
        latest_path = "nkat_mnist_checkpoints/best_model.pt"
        if os.path.exists(latest_path):
            checkpoint = torch.load(latest_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"Loaded latest checkpoint from epoch {epoch}")
        else:
            print("Warning: No checkpoint found, using random initialization")
    
    model.eval()
    return model, config

def create_test_loader(config):
    """テストデータローダーの作成"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(
        root="data", 
        train=False, 
        download=True,
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=512,  # 高速推論用に大きめのバッチサイズ
        shuffle=False,
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True
    )
    
    return test_loader

def evaluate_model_performance(model, test_loader):
    """モデル性能の詳細評価"""
    print("\n=== NKAT-ViT Performance Evaluation ===")
    
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_confidences = []
    inference_times = []
    
    # 混合精度推論とGradScaler
    scaler = torch.cuda.amp.GradScaler()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # 推論時間測定開始
            start_time = time.time()
            
            # 通常推論（混合精度を無効化）
            outputs = model(images)
            # NKAT出力の形式チェック
            if isinstance(outputs, dict):
                # 出力が辞書の場合、利用可能なキーを確認
                if batch_idx == 0:  # 最初のバッチでデバッグ情報を出力
                    print(f"Model output keys: {list(outputs.keys())}")
                
                # 一般的なキー名をチェック
                if 'logits' in outputs:
                    logits = outputs['logits']
                elif 'predictions' in outputs:
                    logits = outputs['predictions'] 
                elif 'output' in outputs:
                    logits = outputs['output']
                else:
                    # キーが見つからない場合は最初の値を使用
                    first_key = list(outputs.keys())[0]
                    logits = outputs[first_key]
                    if batch_idx == 0:
                        print(f"Using key '{first_key}' as logits")
            else:
                logits = outputs
            
            probabilities = F.softmax(logits, dim=1)
            
            # 推論時間測定終了
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # デバッグ情報（最初のバッチのみ）
            if batch_idx == 0:
                print(f"Logits shape: {logits.shape}")
                print(f"Logits sample (first 5): {logits[0].cpu().numpy()}")
                print(f"Logits min/max: {logits.min():.4f} / {logits.max():.4f}")
                print(f"Probabilities sample: {probabilities[0].cpu().numpy()}")
                print(f"Probabilities sum: {probabilities[0].sum().item():.4f}")
            
            # 予測と信頼度
            confidences, predictions = torch.max(probabilities, 1)
            
            # 統計更新
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # 結果保存
            all_preds.append(predictions.cpu())
            all_labels.append(labels.cpu())
            all_confidences.append(confidences.cpu())
    
    # 結果統計
    accuracy = 100.0 * correct / total
    avg_inference_time = np.mean(inference_times)
    throughput = len(test_loader.dataset) / sum(inference_times)
    
    print(f"\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}% ({correct}/{total})")
    print(f"Average inference time per batch: {avg_inference_time:.4f}s")
    print(f"Throughput: {throughput:.2f} samples/sec")
    print(f"RTX3080 Memory Usage: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
    
    # 結果をファイルに保存
    results = {
        "preds": torch.cat(all_preds),
        "labels": torch.cat(all_labels),
        "confidences": torch.cat(all_confidences),
        "accuracy": accuracy,
        "inference_time": avg_inference_time,
        "throughput": throughput
    }
    
    torch.save(results, "analysis/test_results.pt")
    print(f"Results saved to analysis/test_results.pt")
    
    return results

def analyze_class_performance(results):
    """クラス別性能分析"""
    preds = results["preds"].numpy()
    labels = results["labels"].numpy()
    confidences = results["confidences"].numpy()
    
    print("\n=== Class-wise Performance Analysis ===")
    
    # クラス別統計
    class_stats = {}
    for class_id in range(10):
        mask = labels == class_id
        if mask.sum() > 0:
            class_preds = preds[mask]
            class_conf = confidences[mask]
            
            class_acc = (class_preds == class_id).mean() * 100
            avg_conf = class_conf.mean()
            
            class_stats[class_id] = {
                "accuracy": class_acc,
                "confidence": avg_conf,
                "count": mask.sum()
            }
            
            print(f"Class {class_id}: Acc={class_acc:.2f}%, Conf={avg_conf:.4f}, Count={mask.sum()}")
    
    return class_stats

def find_difficult_samples(results, top_k=20):
    """困難サンプルの特定"""
    preds = results["preds"].numpy()
    labels = results["labels"].numpy()
    confidences = results["confidences"].numpy()
    
    # 誤分類サンプル
    wrong_indices = np.where(preds != labels)[0]
    
    if len(wrong_indices) > 0:
        # 信頼度が高いのに間違ったサンプル（最も問題）
        wrong_confidences = confidences[wrong_indices]
        high_conf_wrong = wrong_indices[np.argsort(wrong_confidences)[-top_k:]]
        
        print(f"\n=== Top {min(top_k, len(high_conf_wrong))} High-Confidence Errors ===")
        for i, idx in enumerate(reversed(high_conf_wrong)):
            print(f"Sample {idx}: True={labels[idx]}, Pred={preds[idx]}, Conf={confidences[idx]:.4f}")
    
    return wrong_indices

def generate_performance_report(results, class_stats):
    """性能レポートの生成"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = {
        "timestamp": timestamp,
        "model": "NKAT-Vision Transformer",
        "overall_accuracy": results["accuracy"],
        "throughput_samples_per_sec": results["throughput"],
        "inference_time_per_batch": results["inference_time"],
        "class_performance": class_stats,
        "total_samples": len(results["labels"]),
        "gpu_memory_gb": torch.cuda.max_memory_allocated()/1024**3
    }
    
    # JSONレポート保存
    with open(f"analysis/performance_report_{timestamp}.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nPerformance report saved to analysis/performance_report_{timestamp}.json")
    
    return report

def main():
    """メイン実行関数"""
    print("NKAT-Transformer Final Evaluation")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # モデル読み込み
    model, config = load_best_model()
    
    # テストデータ準備
    test_loader = create_test_loader(config)
    
    # パフォーマンス評価
    results = evaluate_model_performance(model, test_loader)
    
    # クラス別分析
    class_stats = analyze_class_performance(results)
    
    # 困難サンプル特定
    difficult_samples = find_difficult_samples(results)
    
    # レポート生成
    report = generate_performance_report(results, class_stats)
    
    print("\n" + "=" * 50)
    print("Evaluation completed successfully!")
    print(f"Final Test Accuracy: {results['accuracy']:.4f}%")
    print("Next step: Run analyse_confusion.py for detailed error analysis")

if __name__ == "__main__":
    main() 