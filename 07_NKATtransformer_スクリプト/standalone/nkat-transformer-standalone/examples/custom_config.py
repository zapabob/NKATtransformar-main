#!/usr/bin/env python3
"""
Custom Configuration Example
カスタム設定例
"""

from nkat_core_standalone import NKATTrainer, NKATConfig

# カスタム設定クラス
class FastConfig(NKATConfig):
    """高速訓練用設定"""
    def __init__(self):
        super().__init__()
        self.num_epochs = 50
        self.batch_size = 128
        self.learning_rate = 2e-4

class PreciseConfig(NKATConfig):
    """高精度追求用設定"""
    def __init__(self):
        super().__init__()
        self.num_epochs = 200
        self.batch_size = 32
        self.learning_rate = 5e-5
        self.class_weight_boost = 2.0

def main():
    print("🔧 Custom Configuration Examples")
    
    choice = input("設定を選択してください (1: 高速, 2: 高精度): ")
    
    if choice == "1":
        config = FastConfig()
        print("⚡ 高速訓練設定を選択")
    elif choice == "2":
        config = PreciseConfig()
        print("🎯 高精度追求設定を選択")
    else:
        config = NKATConfig()
        print("📋 標準設定を使用")
    
    trainer = NKATTrainer(config)
    accuracy = trainer.train()
    
    print(f"結果: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
