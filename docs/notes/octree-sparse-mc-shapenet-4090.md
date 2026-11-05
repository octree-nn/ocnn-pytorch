# ShapeNet r256：Triton sparse MC 与 cubvh 的真实数据比较

日期：2026-11-05。机器：4090 / SIG4090，使用物理 GPU 4（RTX 4090）。
本轮检查 4 个样本，未修改数据集、原始 OBJ 或已保存的 octree。

## 结果

当前实现与 cubvh 在全部 4 个样本上得到相同顶点数和三角形连接；通过
双向最近顶点匹配确认一一对应，世界坐标最大差均为
`1.1920928955078125e-7`。所有抽出网格的顶点均有限、闭合且面方向一致，
零面积面、边界边、非流形边计数均为 0。本轮没有做自交或顶点流形性审计，
这些检查不能直接证明任意输入下的拓扑保证。

| 样本 | 相交 cells | 三角形数 | 当前实现：octree→mesh | cubvh 核心 | cubvh：octree→mesh |
| --- | ---: | ---: | ---: | ---: | ---: |
| 飞机 `101556558504…` | 47,831 | 95,712 | 1.33 ms | 1.54 ms | 2.12 ms |
| 椅子 `1006be65e7bc…` | 99,113 | 198,280 | 1.96 ms | 2.40 ms | 3.88 ms |
| 汽车 `1005ca47e516…` | 160,086 | 320,876 | 1.33 ms | 3.08 ms | 3.79 ms |
| 修复 SDF 汽车 `88ad6f5df57c…` | 178,895 | 358,932 | 1.35 ms | 3.22 ms | 4.07 ms |

每种路径预热 3 次，随后测量 20 次同步 wall time，表中为中位数。
不计磁盘读写、建树、首次 JIT 或渲染。`octree→mesh` 两列均包含从
octree 取得坐标与角点数据的工作；cubvh 核心一列使用预先准备好的
连续 CUDA int32 coords 和 float32 corners。顶点世界坐标转换不计时。
缓存预热后的 `nempty_index` 对两条路径都可复用。
当前实现的 octree→mesh 在这 4 例中快约 1.60–3.02 倍。

这是共享主机上的小规模检查，计时存在抖动，例如椅子 cubvh 总时间的
20 次样本范围为 3.08–16.44 ms；完整原始计时保存在 `report.json`。
上述比值不代表所有 ShapeNet 样本或其他设备上的稳定加速比。

## 与 dense SDF、原始模型的关系

逐一核对所有相交 cell 的八角值，与原 dense SDF（正向补一层 +1）
完全一致，最大角值误差为 0。将全部输出顶点代入 dense SDF 三线性
插值，最大绝对残差不超过 `5.993e-8`，没有非有限结果。
因此此次抽面复现的是已保存 SDF 的零等值面。

原始模型使用 `ShapeNetCore.v1/<category>/<model>/model.obj`，由
`r256/bbox/<category>/<model>.npz` 中的参数对齐：

```python
center = (bbmin + bbmax) / 2
source_vertices = (source_vertices - center) * (2 * mul / (bbmax-bbmin).max())
extracted_vertices = sample_index_vertices * (2 / 256) - 1
```

四个 OBJ 的 bounds 均与 bbox 元数据匹配。本轮原始源模型采用完整几何，
只在渲染时忽略纹理，未执行焊点、删面、修复或平滑。
每个方向固定种子采样 10,000 个三角形表面点，用 cubvh BVH 查询到另一个
网格三角形的最近距离；这是采样距离，不是精确 Hausdorff 距离。

| 样本 | 抽面→原模型：均值 | 抽面→原模型：P95 | 原模型→抽面：P95 |
| --- | ---: | ---: | ---: |
| 飞机 | 0.007697 | 0.008167 | 0.032671 |
| 椅子 | 0.007688 | 0.007853 | 0.007923 |
| 汽车 | 0.007785 | 0.008380 | 0.019542 |
| 修复 SDF 汽车 | 0.007828 | 0.008786 | 0.013570 |

距离单位为归一化世界坐标，一体素宽为 `2/256=0.0078125`。
原模型与 SDF 表面存在明显可测偏差；可视化也可看到薄结构增厚、局部
缝隙填充和细节变化。两种抽面器得到同一结果，并且 dense SDF 残差很小，
所以不能将这些源模型偏差归因于当前 Triton 迁移。

所有选中角值都不恰好为 0，本轮真实样本不覆盖 cubvh `<0` 与本实现
`<=0` 的分类差异；已有解析零平面测试另行覆盖本实现的零值约定。
汽车 `88ad…` 使用当前已修复的 SDF，而不是重现此前异常的旧 SDF。

## 输入与环境

- Octree：`/data/zhoucz/datasets/ShapeNet/r256/octree-sdf`。
- dense SDF：优先读取 `/data/zhoucz/datasets/ShapeNet/r256/sdf` 中的覆盖文件，
  其余来自 `/data/wangps/datasets/ShapeNet/r256/sdf`。
- 原模型和 bbox：`/data/wangps/datasets/ShapeNet`。
- 样本完整 ID 见脚本 `DEFAULT_SAMPLES` 和结果 `report.json`。
- 保存的 octree 为 version 2，depth=8、full_depth=4、257³ 角点、float32、
  非 adaptive。比较脚本只为抽面读取该格式，不改变原来的查询域语义。
- cubvh 固定提交：`de1badf89b54975d407c42a02a47d41fa149e486`，使用上游
  完整 CUDA 扩展，`iso=0`、`ensure_consistency=False`。
- 计时环境：独立 uv 环境，PyTorch 2.10.0+cu128、Triton 3.6.0、CUDA 12.8，
  cubvh 编译架构 8.9，Eigen 来自 cmeel-eigen 3.4.0.2。
- 渲染环境：独立 uv 环境，pyrender EGL、NumPy 1.26.4；1280×960 单面板，
  缩放排版为正反两个 3 列对照图。每行共享相机、比例、光照和材质。

## 复现与产物

本分支脚本：

```bash
PYTHONPATH=. python projects/tools/compare_sparse_mc.py \
  --dataset-root /data/zhoucz/datasets/ShapeNet/r256 \
  --source-root /data/wangps/datasets/ShapeNet \
  --output /path/to/results --repeats 20

PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=4 python \
  projects/tools/render_sparse_mc_comparison.py /path/to/results
```

远端实验根目录：
`/data/zhoucz/jit3d-scratch/octree-spmc-20261105-01a1a9b9`。
计时环境是 `env128`，渲染环境是 `env`，代码快照在 `code`。

本地完整产物：
`/mnt/sdc/zhoucz/jit3d-scratch/octree-sparse-mc-results/shapenet-4090`。
其中 `report.json` 是完整数值记录；每个样本下含 `source.ply`、
`triton.ply`、`cubvh.ply`、`checks.json` 和各视角高分辨率图片。
`results.tar.gz` 包含上述网格、报告和图片，便于整体取走。
