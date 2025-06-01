# nkat_transformer/__init__.py
"""
🧠 NKAT-Transformer: LLMスタイルハイパーパラメータ対応版
Theory-Practical Equilibrium (TPE) 最適化対応
"""

from .model import NKATVisionTransformer, NKATLightweight, NKATHeavyweight
from .modules.nkat_attention import NKATAttention, NKATTransformerBlock

__version__ = "2.0.0"
__author__ = "NKAT Research Team"

__all__ = [
    'NKATVisionTransformer',
    'NKATLightweight', 
    'NKATHeavyweight',
    'NKATAttention',
    'NKATTransformerBlock'
] 