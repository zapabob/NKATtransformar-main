#!/usr/bin/env python3
"""
Full Training - NKAT-Transformer
本格的な99%+精度訓練
"""

from nkat_core_standalone import NKATTrainer, NKATConfig

def main():
    print("🎯 NKAT-Transformer Full Training")
    print("99%+精度を目指した本格訓練を開始します...")
    
    # 本格設定
    config = NKATConfig()
    config.num_epochs = 100
    config.batch_size = 64
    
    # GPUメモリが不足する場合の調整
    # config.batch_size = 32
    
    print(f"設定:")
    print(f"• エポック数: {config.num_epochs}")
    print(f"• バッチサイズ: {config.batch_size}")
    print(f"• 困難クラス: {config.difficult_classes}")
    
    # 学習実行
    trainer = NKATTrainer(config)
    accuracy = trainer.train()
    
    print(f"\n🎉 訓練完了！")
    print(f"最終精度: {accuracy:.2f}%")
    
    if accuracy >= 99.0:
        print("🏆 99%+達成おめでとうございます！")
    else:
        print("📈 さらなる調整で99%+を目指しましょう")

if __name__ == "__main__":
    main()
