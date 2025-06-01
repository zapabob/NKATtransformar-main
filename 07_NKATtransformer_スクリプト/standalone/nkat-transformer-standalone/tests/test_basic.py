#!/usr/bin/env python3
"""
Test Script - NKAT-Transformer
基本動作テスト
"""

import torch
import sys
import os

# パス追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nkat_core_standalone import NKATConfig, NKATVisionTransformer, load_pretrained

def test_model_creation():
    """モデル作成テスト"""
    print("🧪 Model Creation Test")
    
    config = NKATConfig()
    model = NKATVisionTransformer(config)
    
    # パラメータ数確認
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model created: {total_params:,} parameters")
    
    return model

def test_forward_pass():
    """順伝播テスト"""
    print("🔄 Forward Pass Test")
    
    config = NKATConfig()
    model = NKATVisionTransformer(config)
    
    # ダミー入力
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 28, 28)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✅ Forward pass: Input {dummy_input.shape} → Output {output.shape}")
    print(f"✅ Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    return output

def test_cuda_availability():
    """CUDA動作テスト"""
    print("🎮 CUDA Availability Test")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA version: {torch.version.cuda}")
        print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ CUDA not available - using CPU")

def main():
    print("🚀 NKAT-Transformer Test Suite")
    print("=" * 50)
    
    try:
        # CUDA テスト
        test_cuda_availability()
        print()
        
        # モデル作成テスト
        model = test_model_creation()
        print()
        
        # 順伝播テスト
        output = test_forward_pass()
        print()
        
        print("🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
