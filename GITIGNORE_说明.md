# .gitignore 设计说明

## 设计原则

1. **只上传核心代码**：保留所有源代码和配置文件
2. **保留目录结构**：使用 `.gitkeep` 文件保留重要目录的层级结构
3. **忽略大文件**：模型文件、数据集、视频等大文件不上传，保存在本地

## 会被 Git 跟踪的内容（核心代码）

✅ **源代码目录**
- `hmr4d/` - 核心代码
- `tools/` - 工具脚本
- `docs/` - 文档
- `third-party/` - 第三方库代码

✅ **配置文件**
- `setup.py`
- `requirements.txt`
- `pyproject.toml`
- `README.md`
- `LICENSE`

✅ **目录结构占位文件**
- `inputs/checkpoints/.gitkeep` - 保留模型目录结构
- `inputs/videos/.gitkeep` - 保留视频目录结构
- `outputs/.gitkeep` - 保留输出目录结构
- `inputs/3DPW/.gitkeep` 等 - 保留数据集目录结构

## 会被 Git 忽略的内容（不上传）

❌ **模型文件**（inputs/checkpoints/ 下）
- `*.ckpt` - PyTorch Lightning checkpoint
- `*.pth` - PyTorch 模型
- `*.pt` - PyTorch 模型
- `*.pkl` - Pickle 文件
- `*.npz` - NumPy 压缩文件
- `*.h5`, `*.hdf5` - HDF5 文件
- `*.safetensors` - SafeTensors 格式

❌ **数据集**（inputs/ 下）
- `3DPW/`, `AMASS/`, `BEDLAM/` 等数据集目录的内容
- `*.tar.gz`, `*.tar`, `*.zip` 压缩包

❌ **输入视频**（inputs/videos/ 下）
- `*.mp4`, `*.avi`, `*.mov`, `*.mkv` 等视频文件

❌ **输出结果**（outputs/ 下）
- 所有运行结果和生成的文件

❌ **临时文件**
- `__pycache__/` - Python 缓存
- `*.pyc` - 编译的 Python 文件
- `*.egg-info/` - 包信息
- `*.log` - 日志文件
- `.pytest_cache/` - 测试缓存
- `wandb/`, `lightning_logs/` - 训练日志

## 目录结构示例

上传到 Git 后的目录结构：

```
GVHMR/
├── hmr4d/              ✅ 核心代码
├── tools/               ✅ 工具脚本
├── docs/                ✅ 文档
├── third-party/         ✅ 第三方库
├── inputs/
│   ├── checkpoints/
│   │   ├── .gitkeep    ✅ 保留目录结构
│   │   ├── body_models/
│   │   │   └── .gitkeep ✅
│   │   ├── gvhmr/      ✅ 目录存在
│   │   │   └── (模型文件被忽略)
│   │   └── ...
│   ├── videos/
│   │   ├── .gitkeep    ✅ 保留目录结构
│   │   └── (视频文件被忽略)
│   └── 3DPW/
│       └── .gitkeep    ✅ 保留目录结构
├── outputs/
│   └── .gitkeep        ✅ 保留目录结构
├── setup.py            ✅
├── requirements.txt    ✅
└── README.md           ✅
```

## 使用方法

1. **首次克隆后**，需要从公司公用硬盘复制模型文件到 `inputs/checkpoints/`
2. **目录结构已保留**，可以直接使用，无需手动创建目录
3. **运行代码时**，模型文件会自动从本地读取

## 注意事项

- `.gitkeep` 文件是空文件，仅用于保留目录结构
- 如果需要在某个子目录也保留结构，可以创建对应的 `.gitkeep` 文件
- 模型文件路径在代码中是硬编码的，确保本地路径与代码中的路径一致
