# DS21-NeRF: (并不) Advanced 的 NeRF

> 清华大学自动化系数据结构课程（20 秋）科研大作业 C

此项目基于 **Ne**ural **R**adiance **F**ield 的不完全代码（PyTorch 版本）：

- 通过实现层次化采样提升训练的效率
- 实现从 NeRF 中重建 3D 场景
- 通过八叉树数据结构为 3D 重建提速

此 Repo 中的代码由 [Aiden Li](https://github.com/YuyangLee) 补全，欢迎讨论交流，拒绝抄袭。

## 修改内容

基于课程作业要求，对代码做出了如下调整：

- 将部分代码（光线计算、采样、可视化等）提取封装成为独立模块，调整优化了变量命名与代码结构
- 优化了部分代码风格
- 编写了层次化采样（见 `run_nerf.py` 第 375 行左右）
    - 依赖从 piecewise-constant PDF 逆变换采样的实现（见 `utils/sampling.py`）
- 编写了用于 3D 重建的类：
    - Scene 类（见 `models/scene.py`）
    - OCTreeNode 类（见 `models/octree.py`）
    - 轻量级 OCTree（见 `models/octree_lite.py`）
    - Volume 类（见 `models/volume.py`）
- 编写了 3D 重建的三种方案（见 `models/scene.py`）：
    - 直接重建
    - 基于 OCTree 重建
    - 基于轻量级 OCTree 重建
- 编写了基于 `plotly.graph_object` 的体素可视化（见 `utils/visualization.py`）
- 编写了 3D 重建 bestbench（见 `run_nerf.py` 末尾）
- ...

# DS21-NeRF: (Not That) Advanced NeRF

> The term project C for Data Structure course, Fall 2021, Tsinghua University

In this project we, based on the incomplete code of **Ne**ural **R**adiance **F**ield,

- boost the training by implementing the **hierarchical sampling** in the paper,
- implement the 3D reconstruction for the trained scene,
- and boost the reconstruction using OCTree.

The code in this repository is based on a incomplete PyTorch version of NeRF and furtherimplemented by [Aiden Li](https://github.com/YuyangLee). Discussion and healthy arguments are welcome but **any plagiarism is forbidden**.

