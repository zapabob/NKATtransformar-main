#!/usr/bin/env python3
"""
Quick Demo - NKAT-Transformer
クイックデモ（5エポック軽量版）
"""

from nkat_core_standalone import NKATTrainer, NKATConfig

def main():
    print("🚀 NKAT-Transformer Quick Demo")
    print("5エポックの軽量デモを実行します...")
    
    # 軽量設定
    config = NKATConfig()
    config.num_epochs = 5
    config.batch_size = 32
    
    # 学習実行
    trainer = NKATTrainer(config)
    accuracy = trainer.train()
    
    print(f"\n✅ デモ完了！")
    print(f"精度: {accuracy:.2f}%")
    print("本格的な99%+学習はnum_epochs=100で実行してください")

if __name__ == "__main__":
    main()
