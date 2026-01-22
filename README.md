# GVHMR: World-Grounded Human Motion Recovery via Gravity-View Coordinates
### [Project Page](https://zju3dv.github.io/gvhmr) | [Paper](https://arxiv.org/abs/2409.06662)

> World-Grounded Human Motion Recovery via Gravity-View Coordinates  
> [Zehong Shen](https://zehongs.github.io/)<sup>\*</sup>,
[Huaijin Pi](https://phj128.github.io/)<sup>\*</sup>,
[Yan Xia](https://isshikihugh.github.io/scholar),
[Zhi Cen](https://scholar.google.com/citations?user=Xyy-uFMAAAAJ),
[Sida Peng](https://pengsida.net/)<sup>†</sup>,
[Zechen Hu](https://zju3dv.github.io/gvhmr),
[Hujun Bao](http://www.cad.zju.edu.cn/home/bao/),
[Ruizhen Hu](https://csse.szu.edu.cn/staff/ruizhenhu/),
[Xiaowei Zhou](https://xzhou.me/)  
> SIGGRAPH Asia 2024

<p align="center">
    <img src=docs/example_video/project_teaser.gif alt="animated" />
</p>

## 环境安装（Installation）

> **提示**：下方是官方提供的安装步骤，但不够完整，安装过程中如果遇到缺失的包，根据 Cursor 或错误提示进行修复即可。

### 1. 环境配置

```bash
git clone https://github.com/zju3dv/GVHMR
cd GVHMR

conda create -y -n gvhmr python=3.10
conda activate gvhmr
pip install -r requirements.txt
pip install -e .
```

> 提示：如果要在其他仓库中以可编辑模式安装 gvhmr，可以在 settings.json 中添加 `"python.analysis.extraPaths": ["path/to/your/package"]`

### 2. 可选：DPVO 安装（不推荐，会影响推理速度）

如果需要使用 DPVO 视觉里程计（通过 `--use_dpvo` 参数），需要额外安装：

```bash
cd third-party/DPVO
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty && rm -rf eigen-3.4.0.zip
pip install torch-scatter -f "https://data.pyg.org/whl/torch-2.3.0+cu121.html"
pip install numba pypose
export CUDA_HOME=/usr/local/cuda-12.1/
export PATH=$PATH:/usr/local/cuda-12.1/bin/
pip install -e .
```

### 3. 创建输入输出目录

```bash
mkdir inputs
mkdir outputs
```

### 4. 下载模型权重（Checkpoints）

#### 4.1 SMPL/SMPLX 模型（必需）

需要从官方网站注册下载：
- [SMPL](https://smpl.is.tue.mpg.de/)
- [SMPLX](https://smpl-x.is.tue.mpg.de/)

下载后按以下结构放置：

```
inputs/checkpoints/
├── body_models/smplx/
│   └── SMPLX_{GENDER}.npz  # SMPLX（用于预测和评估）
└── body_models/smpl/
    └── SMPL_{GENDER}.pkl   # SMPL（用于渲染和评估）
```

#### 4.2 预训练模型（已经下载好了，在公司公用硬盘里）

从 [Google Drive](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD?usp=drive_link) 下载预训练模型（下载即表示同意相应许可协议）：

```bash
mkdir -p inputs/checkpoints
```

下载后应包含以下文件：

```
inputs/checkpoints/
├── dpvo/
│   └── dpvo.pth                    # DPVO 视觉里程计模型（可选）
├── gvhmr/
│   └── gvhmr_siga24_release.ckpt  # GVHMR 主模型（必需）
├── hmr2/
│   └── epoch=10-step=25000.ckpt   # HMR2 模型
├── vitpose/
│   └── vitpose-h-multi-coco.pth   # ViTPose 关键点检测模型
└── yolo/
    └── yolov8x.pt                  # YOLO 人体检测模型
```

### 5. 下载数据集（可选，仅用于训练和测试。已经下载好了，在公司公用硬盘里）

> **注意**：我们只提供预处理后的数据，不提供原始数据集。你需要从原始网站下载原始数据（标注、视频等）。由于许可限制，我们无法提供原始数据。下载预处理数据即表示同意原始数据集的条款，仅用于研究目的。

从 [Google Drive](https://drive.google.com/drive/folders/10sEef1V_tULzddFxzCmDUpsIqfv7eP-P?usp=drive_link) 下载预处理数据，放置在 `inputs` 文件夹中：

```bash
cd inputs

# 训练数据集
tar -xzvf AMASS_hmr4d_support.tar.gz
tar -xzvf BEDLAM_hmr4d_support.tar.gz
tar -xzvf H36M_hmr4d_support.tar.gz

# 测试数据集
tar -xzvf 3DPW_hmr4d_support.tar.gz
tar -xzvf EMDB_hmr4d_support.tar.gz
tar -xzvf RICH_hmr4d_support.tar.gz
```

解压后的目录结构应为：

```
inputs/
├── AMASS/hmr4d_support/
├── BEDLAM/hmr4d_support/
├── H36M/hmr4d_support/
├── 3DPW/hmr4d_support/
├── EMDB/hmr4d_support/
└── RICH/hmr4d_support/
```

> **提示**：如果只是使用 Demo 功能处理视频，**不需要下载训练/测试数据集**，只需要下载模型权重即可。

### 6. 项目目录结构说明

为了正确放置模型文件和运行项目，请确保目录结构如下：

#### inputs/ 目录结构（必须是这样的结构层级才行！！！）

```
inputs/
├── checkpoints/                    # 模型权重目录（必需）
│   ├── body_models/                 # SMPL/SMPLX 人体模型
│   │   ├── smpl/
│   │   │   ├── SMPL_NEUTRAL.pkl     #注意，这不是SMPLX官网的，是SMPL官网的下载文件，而且下载好了必须改名为SMPL_NEUTRAL.pkl
│   │   │   ├── SMPL_MALE.pkl
│   │   │   └── SMPL_FEMALE.pkl
│   │   └── smplx/
│   │       ├── SMPLX_NEUTRAL.npz（必须）
│   │       ├── SMPLX_MALE.npz（非必须）
│   │       └── SMPLX_FEMALE.npz（非必须）
│   ├── gvhmr/                       # GVHMR 主模型（必需）
│   │   └── gvhmr_siga24_release.ckpt
│   ├── hmr2/                        # HMR2 模型
│   │   └── epoch=10-step=25000.ckpt
│   ├── vitpose/                     # ViTPose 关键点检测模型
│   │   └── vitpose-h-multi-coco.pth
│   ├── yolo/                        # YOLO 人体检测模型
│   │   └── yolov8x.pt
│   └── dpvo/                        # DPVO 视觉里程计模型（可选）
│       └── dpvo.pth
│
├── videos/                          # 输入视频目录
│   ├── test_video.mp4
│   └── ...                          # 其他视频文件
│
└── [数据集目录]                      # 仅用于训练/测试（可选）
    ├── 3DPW/hmr4d_support/
    ├── AMASS/hmr4d_support/
    ├── BEDLAM/hmr4d_support/
    ├── H36M/hmr4d_support/
    ├── EMDB/hmr4d_support/
    └── RICH/hmr4d_support/
```

**关键模型文件路径**：
- GVHMR 主模型：`inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt` ⭐ **必需**
- SMPL 模型：`inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl` ⭐ **必需**
- SMPLX 模型：`inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz` ⭐ **必需**

#### outputs/ 目录结构（输出结果）

```
outputs/
├── demo/                            # 默认输出目录
│   └── {video_name}/               # 以视频名命名的子目录
│       ├── 0_input_video.mp4       # 原始输入视频
│       ├── 1_incam.mp4             # 相机坐标系渲染结果
│       ├── 2_global.mp4            # 世界坐标系渲染结果
│       ├── {video_name}_3_incam_global_horiz.mp4  # 合并对比视频
│       ├── smplx_params.npz         # SMPL-X 参数（用于机器人重定向）⭐
│       ├── hmr4d_results.pt        # 完整预测结果（PyTorch 格式）
│       └── preprocess/             # 预处理中间结果
│           ├── bbx.pt              # 检测框
│           ├── vitpose.pt          # 2D 关键点
│           ├── vit_features.pt     # 视频特征
│           └── slam_results.pt    # 视觉里程计结果（如果使用）
│
└── result_test_XX/                  # 自定义输出目录（通过 --output_root 指定）
    └── {video_name}/
        └── ...                      # 同上结构
```

**重要输出文件**：
- `smplx_params.npz` - 包含 `body_pose`, `global_orient`, `transl`, `betas`, `gender` 等参数，用于机器人重定向 ⭐

> **注意**：
> - 模型文件（checkpoints）保存在本地，不会上传到 Git 仓库
> - 目录结构通过 `.gitkeep` 文件保留，确保克隆后目录存在
> - 首次使用需要从公司公用硬盘复制模型文件到对应目录

## 项目功能说明（Training, Testing, Demo）

### 核心概念：训练 → 测试 → 应用

GVHMR 项目包含三个主要功能模块，它们的关系如下：

```
┌─────────────────────────────────────────────────────────┐
│  1. 训练（Training）- 生成模型权重文件                    │
│     python tools/train.py exp=gvhmr/mixed/mixed         │
│     ↓                                                   │
│     生成 checkpoint: outputs/.../epoch=XXX.ckpt         │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  2. 测试（Testing）- 评估模型性能                         │
│     python tools/train.py task=test ckpt_path=...       │
│     ↓                                                   │
│     输出评估指标（MPJPE等）                               │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  3. 视频处理（Demo）- 实际应用                            │
│     python tools/demo/demo.py --video ...               │
│     ↓                                                   │
│     使用 checkpoint 进行推理                             │
│     ↓                                                   │
│     输出 smplx_params.npz（用于机器人重定向等）           │
└─────────────────────────────────────────────────────────┘
```

### 1. 训练（Training）- `tools/train.py task=fit`

**作用**：训练模型，生成 checkpoint 文件（.ckpt）

**命令**：
```bash
python tools/train.py exp=gvhmr/mixed/mixed
```

**训练过程**：
- 使用训练数据集：AMASS、BEDLAM、H36M、3DPW 等
- 训练模型参数（神经网络权重）
- 自动保存 checkpoint：
  - 每 10 个 epoch 保存一次
  - 保存在 `outputs/${data_name}/${exp_name}/` 目录下
  - 文件名类似：`epoch=10-step=25000.ckpt`、`last.ckpt` 等

**训练输出**：
- 模型 checkpoint 文件（.ckpt）- **这是你需要的模型权重文件**
- 训练日志（TensorBoard）

**注意**：训练需要大量数据和计算资源（官方模型在 2x4090 上训练了 420 个 epoch）

### 2. 测试（Testing）- `tools/train.py task=test`

**作用**：在测试数据集上评估模型性能

**命令**：
```bash
# 使用官方预训练模型测试
python tools/train.py global/task=gvhmr/test_3dpw_emdb_rich exp=gvhmr/mixed/mixed ckpt_path=inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt

# 使用自己训练的模型测试
python tools/train.py global/task=gvhmr/test_3dpw_emdb_rich exp=gvhmr/mixed/mixed ckpt_path=outputs/your_exp/your_checkpoint.ckpt
```

**测试过程**：
- 在测试数据集上运行模型（3DPW、RICH、EMDB）
- 计算评估指标（MPJPE、PA-MPJPE 等）
- 输出测试报告

**测试输出**：
- 评估指标（用于论文或对比）

### 3. 视频处理（Demo）- `tools/demo/demo.py`

**作用**：使用 checkpoint 对视频进行推理，生成人体运动参数

**命令**：
```bash
# 使用官方预训练模型（默认，推荐）
python tools/demo/demo.py --video inputs/videos/test_video.mp4 -s

# 使用自己训练的模型（需要修改配置文件或通过 Hydra 覆盖）
# 方法1：修改 hmr4d/configs/demo.yaml 中的 ckpt_path
# 方法2：通过 Hydra 覆盖配置（需要查看 demo.py 是否支持）
```

**Demo 过程**：
- 加载 checkpoint（默认：`inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt`）
- 预处理视频（检测、关键点提取、特征提取、视觉里程计等）
- 使用模型进行推理
- 生成 SMPL-X 参数和可视化视频

**Demo 输出**：
- `smplx_params.npz` - SMPL-X 参数（用于机器人重定向）
- `hmr4d_results.pt` - 完整的预测结果（PyTorch 格式）
- `1_incam.mp4` - 相机坐标系下的渲染结果
- `2_global.mp4` - 世界坐标系下的渲染结果
- `test_video_3_incam_global_horiz.mp4` - 合并后的对比视频

### 使用场景建议

1. **快速开始（推荐）**：
   - 直接使用官方预训练模型：`python tools/demo/demo.py --video ... -s`
   - 无需训练，直接处理视频

2. **评估模型性能**：
   - 使用测试功能在标准数据集上评估模型

3. **训练自己的模型**：
   - 如果有自己的数据集或需要针对特定场景优化
   - 训练后使用自己的 checkpoint 进行视频处理

4. **机器人重定向应用**：
   - 使用 Demo 功能处理视频
   - 提取 `smplx_params.npz` 文件
   - 将参数映射到机器人关节（需要额外的重定向代码）

### 关键文件说明

- **Checkpoint 文件（.ckpt）**：包含训练好的模型权重
  - 官方预训练：`inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt`
  - 自己训练：`outputs/${data_name}/${exp_name}/epoch=XXX.ckpt`

- **配置文件**：
  - Demo 配置：`hmr4d/configs/demo.yaml`（包含默认 checkpoint 路径）
  - 训练配置：`hmr4d/configs/exp/gvhmr/mixed/mixed.yaml`


## Demo 命令速查（可直接复制粘贴）

> 说明：`--output_root` 默认是 `outputs/demo`，输出会落在 `${output_root}/${video_name}/` 下面。

### 单个视频

```bash
# 1) 静态相机（推荐先跑通；跳过 VO/SLAM）
python tools/demo/demo.py --video inputs/videos/test_video.mp4 -s

# 2) 静态相机 + 自定义输出目录（方便做多次实验，基本上用这个指令就可以了！！！）
python tools/demo/demo.py --video inputs/videos/test_video.mp4 -s --output_root outputs/result_test_01

# 3) 静态相机 + 保存更多中间可视化（更慢/更占空间）
python tools/demo/demo.py --video inputs/videos/test_video.mp4 -s --verbose

# 4) 非静态相机（默认用 SimpleVO；不加 -s 即可）
python tools/demo/demo.py --video inputs/videos/test_video.mp4

# 5) 非静态相机 + 指定全画幅等效焦距（mm），例如 24mm
python tools/demo/demo.py --video inputs/videos/test_video.mp4 --f_mm 24

# 6) 使用 DPVO（可选；通常更慢，且依赖 DPVO 环境）
python tools/demo/demo.py --video inputs/videos/test_video.mp4 --use_dpvo
```

### 批量跑一个文件夹（目录下所有 .mp4/.MP4）

```bash
# 7) 文件夹 + 静态相机
python tools/demo/demo_folder.py -f inputs/videos -d outputs/demo_folder_out -s

# 8) 文件夹（非静态相机；默认 SimpleVO）
python tools/demo/demo_folder.py -f inputs/videos -d outputs/demo_folder_out
```
