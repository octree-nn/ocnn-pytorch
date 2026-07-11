import ast

if not hasattr(ast, 'Num'):
  r''' Triton 3.5.1 constructs ast.Num nodes when compiling range loops.

  The Triton kernels contain loops like:
  ```python
      for k in range(num_k * V):
  ```

  When Triton compiles that JIT kernel, Triton 3.5.1 internally still calls
  `ast.Num(0)` / `ast.Num(1)` while lowering `range(...)`. But in Python 3.14,
  `ast.Num` has been removed; numeric constants now use `ast.Constant`. So the
  kernel compile crashes with:

  ```
      AttributeError: module 'ast' has no attribute 'Num'
  ```

  To fix this, we monkey-patch `ast.Num` to `ast.Constant` for Python 3.14+.
  '''
  ast.Num = ast.Constant


from .conv_fwd_implicit_gemm_splitk import conv_fwd_implicit_gemm_splitk
from .conv_bwd_implicit_gemm_splitk import conv_bwd_implicit_gemm_splitk
from .conv_bwd_implicit_gemm import conv_bwd_implicit_gemm
from .conv_fwd_implicit_gemm import conv_fwd_implicit_gemm

__all__ = [
    'conv_fwd_implicit_gemm_splitk',
    'conv_bwd_implicit_gemm_splitk',
    'conv_bwd_implicit_gemm',
    'conv_fwd_implicit_gemm',
]

from .autotuner import load_autotune_cache
load_autotune_cache()
