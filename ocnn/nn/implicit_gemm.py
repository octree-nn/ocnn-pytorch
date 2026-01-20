import torch
from flex_gemm.kernels.triton.spconv.sparse_submanifold_conv_fwd_implicit_gemm_splitk import (
  sparse_submanifold_conv_fwd_implicit_gemm_splitk,
)
from flex_gemm.kernels.triton.spconv.sparse_submanifold_conv_bwd_implicit_gemm_splitk import (
  sparse_submanifold_conv_bwd_weight_implicit_gemm_splitk,
)


def flex_gemm_forward_implicit(
  data: torch.Tensor,
  weight: torch.Tensor,
  bias: torch.Tensor,
  neighbour: torch.Tensor,
):
  return sparse_submanifold_conv_fwd_implicit_gemm_splitk(
    data, weight, bias, neighbour
  )


def flex_gemm_backward_weight_implicit(
  grad: torch.Tensor, data: torch.Tensor, neighbour: torch.Tensor
):
  return sparse_submanifold_conv_bwd_weight_implicit_gemm_splitk(
    grad, data, neighbour
  )
