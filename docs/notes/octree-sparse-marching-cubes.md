# 非 adaptive octree SDF 的 Sparse Marching Cubes

实现位于 `ocnn/octree/marching_cubes.py`，可选 Triton kernels 位于
`ocnn/octree/_mc_triton.py`。不需要安装 cubvh 或编译 C++ 扩展。
经典 MC 查表来自 cubvh 的固定提交
`de1badf89b54975d407c42a02a47d41fa149e486`，保留了上游 MIT 许可。

## 使用

```python
from ocnn.octree import octree_marching_cubes

# tree 已由 build_octree_from_sdf(..., compress=False) 构建。
# 自动在 CUDA + Triton 环境选择 Triton，否则使用 PyTorch。
meshes = octree_marching_cubes(tree, backend='auto', world=True)
vertices, faces = meshes[0]
```

始终返回每个 batch 一个 `OctreeMesh(vertices, faces)`，包括空 mesh。
顶点为 float32，面索引为 int64，与 SDF fields 位于同一 device。
默认 `world=False`，坐标单位为角点采样索引；`world=True` 使用
`vertices * (2 / 2**depth) - 1` 映射到 `[-1,1]`。
其他包围盒可由调用者对索引坐标执行自己的仿射变换。

从格式兼容的已保存 SDF octree 导出 OBJ：

```bash
PYTHONPATH=. python projects/tools/extract_sdf_mesh.py \
  input.pt output.obj --device cuda:0 --backend triton
```

默认导出世界坐标，传 `--sample-coordinates` 保留采样索引。
多 batch 输出为 `output-b000.obj` 等文件；不会合并不同 batch 的边。
命令输出抽面时间（第一次运行包含编译，不包含读盘与 OBJ 写入）。

## 算法与范围

1. 选取最细层 nonempty cells，保留其完整 slot 索引和整数原点。
2. 从每个父节点的 27 点块按 `base+[0,9,12,3,1,10,13,4]`
   gather 八角值，计算 case 和面数。
3. 用 PyTorch 前缀和确定相交边实例及三角形的紧凑写入位置。
4. 为相交边构造 `((x*S+y)*S+z)*3+axis` 的 int64 key，`S=2**depth+1`。
   用 `torch.unique` 和 inverse map 为同一网格边分配唯一顶点。
5. 每条唯一边只插值一次；使用低坐标到高坐标的统一端点方向。
6. 查表生成三角形并映射到唯一顶点，普通构型面朝向正 SDF 一侧。

Triton 实现 gather/case、边实例生成、唯一边插值和面写入；全局前缀和、
去重和代表 cell 选择由 PyTorch 完成。`backend='torch'` 是 CPU/CUDA
参考实现。不会调用 `to_sdf()` 或按全域体积构造数组；工作集随候选
cells、相交边实例和输出大小增长。当前仍有 `[K,8]` 角值和 `[K,12]`
边 mask/map 临时张量，未做极限显存优化。

只支持零等值面、未剪枝、未量化的 float32 SDF octree，深度不超过 16。
共享角点必须一致，输入需为有限值。精确零归入非正侧，与树构建中的
`sdf>0` 分区一致；这一点有意不同于 cubvh 的严格 `<iso` 分类。
不能仅在这里改 iso 参数来提取其他等值面，因为树未必保存对应 cells。
通过本分支 `prune()` 剪枝的对象会被拒绝，device 转换保留该标记。
外部构造或旧版对象仍由调用者保证为非 adaptive，不执行全树几何审计。

实现保留经典 MC 的退化面，不包含 MC33、消歧、修复、细粗交界处理或
autograd。动态输出分配和全局去重可引入 GPU 同步，暂不支持完整的
`torch.compile(fullgraph=True)` 或 CUDA graph capture。

## 验证

`test/test_marching_cubes.py` 覆盖上边界平面、世界坐标、法向方向、
球体与 scikit-image classic MC 的独立对照、共享边闭合性、精确零值、
空场、batch 隔离和不支持的输入，并在 CUDA 上对比 PyTorch/Triton。
与 scikit-image 对照时对齐 MC 的 xyz 轴，而不是只交换最终输出坐标；
查表剖分不保证在轴置换后选择同一条对角线。

```bash
OMP_NUM_THREADS=4 PYTHONPATH=. python -m unittest \
  test.test_marching_cubes test.test_octree test.test_sdf_io -v
```

本工作树基于 `a8b9364`，并复制创建时主工作区的未提交 SDF 修正作为
基础。原工作区未移动或删除文件；本次功能的新增文件应与这些继承修改区分。

### 2026-11-05 验证记录

上述 20 项测试全部通过，包含真实 RTX 3090 上的 CUDA/Triton 检查。
测试使用 uv 创建的独立 Python 3.10 环境，复用本机依赖缓存：
PyTorch 2.13.0+cu132、Triton 3.7.1、scikit-image 0.25.2。
GPU 检查在沙箱外运行；沙箱内不能访问 NVIDIA 驱动。

另对偏心解析球体进行 5 次同步计时，预先运行一次排除编译与 warmup，
不包含 dense 场生成、建树、读写盘；下面是 API 整体调用的中位数，
包含 Python 调度、去重和世界坐标转换前的完整抽面流程。
这只是当前机器的初步小样本结果，未与 cubvh 比较。

| depth | 相交 cells | 顶点 / 三角形 | PyTorch CUDA | Triton + PyTorch |
| --- | ---: | ---: | ---: | ---: |
| 5 | 672 | 670 / 1,336 | 25.43 ms | 3.51 ms |
| 7 | 10,632 | 10,630 / 21,256 | 23.98 ms | 16.19 ms |
| 8 | 42,520 | 42,518 / 85,032 | 25.71 ms | 16.28 ms |

三个尺度的两种后端顶点与面均通过一致性检查。
原始计时和样例输出位于
`/mnt/sdc/zhoucz/jit3d-scratch/octree-sparse-mc-results/`。
