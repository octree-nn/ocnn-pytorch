import os

allow_tf32 = os.getenv('OCNN_ALLOW_TF32', '1') == '1'
AUTOSAVE_AUTOTUNE_CACHE = os.getenv('OCNN_AUTOSAVE_AUTOTUNE', '1') == '1'
AUTOTUNE_CACHE_PATH = os.getenv('OCNN_AUTOTUNE_CACHE_PATH',
                                os.path.expanduser('~/.ocnnconvt/autotune_cache.json'))

from .conv_fwd_implicit_gemm_splitk import conv_fwd_implicit_gemm_splitk
from .conv_bwd_implicit_gemm_splitk import conv_bwd_implicit_gemm_splitk

__all__ = [
    'conv_fwd_implicit_gemm_splitk',
    'conv_bwd_implicit_gemm_splitk',
]

from .autotuner import load_autotune_cache
load_autotune_cache()
