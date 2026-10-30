# LLM from Scratch

从零实现一个基于 PyTorch 的小型语言模型，记录从 BPE Tokenizer 到 Transformer、训练流程和文本生成的学习与实践过程。

本项目参考 Stanford CS336 等相关学习资料，尝试将语言模型的核心组件拆解，并尽可能独立实现。项目的重点不只是让模型运行起来，更是理解每个组件的原理、数据流与实现细节。

该项目仅供学习使用，不具有工业生产能力

## 1. 项目背景

最初开始这个项目时，我对大语言模型的理解主要停留在概念和公式层面。

为了真正理解 LLM 的工作过程，我决定从基础组件开始，逐步实现一个完整的语言模型。

开发过程中，我遇到过不少问题：BPE 训练速度过慢、张量维度不匹配、模型 Loss 无法正常收敛，以及不同模块之间的数据接口问题。

这些问题促使我进一步学习数据结构、算法优化、张量运算和模型训练流程。相比直接调用现成框架，我更希望通过自己实现、调试和修改代码，建立对语言模型内部机制的理解。

## 2. 项目概览

目前项目主要包含以下模块：

| 模块 | 实现内容 |
|---|---|
| Tokenizer | BPE 训练、文本编码与解码 |
| Embedding | Token Embedding |
| Attention | 因果自注意力机制 |
| Position Encoding | RoPE |
| Normalization | RMSNorm |
| Feed Forward | SwiGLU |
| Transformer | Transformer Decoder |
| Optimizer | AdamW |
| Loss | Cross Entropy |
| Training | 梯度裁剪、学习率调度 |
| Generation | Temperature、Top-p 采样、自回归生成 |

## 3. 项目结构

```text
llm-from-scratch/
├── src/
│   └── llmfs/
│       ├── cli.py       # llmfs 命令入口
│       ├── commands/    # 参数解析、preset 与配置传递
│       ├── configs/     # dataclass 配置和请求定义
│       ├── tokenizer/   # BPE 训练与文本编码、解码
│       ├── model/       # 模型组件与 Transformer
│       ├── training/    # 训练循环、优化器与检查点
│       └── generation/  # 文本生成与采样
├── data/          # 通过 .gitkeep 保留目录，实际数据已被 Git 忽略
│   ├── corpus/       # 原始语料
│   ├── bpe_train/    # BPE 词表与合并规则
│   ├── tokenizer/    # 编码后的 Token ID 数据
│   ├── checkpoints/  # 训练检查点
│   └── model_params/ # 最终模型权重
├── tests/         # 单元测试
├── pyproject.toml
└── README.md
```

Git 通过各目录中的空 `.gitkeep` 文件保留上述五个数据目录。语料、编码后的 `.npy`、词表、检查点和模型权重需要在本地准备或生成，不随源码提交。

## 4. 核心实现

### 4.1 BPE Tokenizer

实现了基于 Byte Pair Encoding（BPE）的分词器，包括：

- 词表初始化与构建
- 相邻 Token Pair 统计
- 高频 Pair 合并
- Tokenizer 编码与解码
- 词表与训练结果保存

在最初的实现中，BPE 训练需要反复扫描语料并统计相邻 Token Pair，处理较大数据集时效率较低。

为此，我尝试引入堆、倒排索引和增量更新等方法，减少不必要的重复计算。同时，也对数据读取方式进行了调整。

在 2GB TinyStories 数据集上的一次测试中，优化后的训练耗时约为 26 秒。具体性能仍与运行环境、参数设置和测试条件有关。

### 4.2 Transformer

模型采用 Decoder-only Transformer 架构，实现了以下组件：

- Token Embedding
- Causal Self-Attention
- Rotary Positional Embedding（RoPE）
- RMSNorm
- SwiGLU Feed Forward Network
- Transformer Decoder Block
- Language Model Head

通过组合上述组件，构建完整的语言模型前向计算流程。

### 4.3 模型训练

训练部分实现了语言模型所需的基础组件：

- Cross Entropy Loss
- AdamW Optimizer
- Gradient Clipping
- Learning Rate Scheduling
- 模型参数更新

训练过程以自回归语言建模为目标，通过预测下一个 Token 学习文本中的统计规律。

### 4.4 文本生成

实现了基于模型输出 Logits 的自回归生成流程，支持：

- Temperature Scaling
- Top-p Sampling
- 逐 Token 生成

通过调整采样参数，可以观察不同生成策略对输出文本的影响。

## 5. 工程实践

除了模型本身的实现，也对项目工程结构进行了整理：

- 使用 dataclass 管理配置与请求参数
- 使用 argparse 构建命令行接口
- 将模型、Tokenizer、训练逻辑和配置进行模块化组织
- 使用标准库 unittest 编写测试，也可以通过 pytest 运行
- 通过 `pyproject.toml` 管理项目及依赖

项目提供统一的命令行入口 `llmfs`，用于执行不同任务。

## 6. 安装

要求 Python 3.10 或更高版本。

克隆或下载项目后，在项目根目录、准备使用的 Python 环境中安装：

```powershell
python -m pip install -e .
```

项目使用 CUDA 时，需要当前环境中的 PyTorch 支持 CUDA。下面的训练和生成示例使用 `--device cuda`；使用 CPU 时改为 `--device cpu`。

## 7. 使用

以下命令均在项目根目录执行。PowerShell 多行命令使用反引号续行，反引号后不要加空格。

### 7.1 准备语料并训练 BPE

自行准备 UTF-8 格式的语料文件，放到 `data/corpus/TinyStoriesV2-GPT4-train.txt`。示例使用 32,000 个 token 的词表。

```powershell
llmfs bpe-train `
  --input-path data/corpus/TinyStoriesV2-GPT4-train.txt `
  --vocab-size 32000
```

输出文件：

```text
data/bpe_train/TinyStoriesV2-GPT4-train/vocab.pkl
data/bpe_train/TinyStoriesV2-GPT4-train/merges.pkl
```

### 7.2 将语料编码为训练数据

```powershell
llmfs tokenize train_mode `
  --vocab-file data/bpe_train/TinyStoriesV2-GPT4-train/vocab.pkl `
  --merges-file data/bpe_train/TinyStoriesV2-GPT4-train/merges.pkl `
  --text_path data/corpus/TinyStoriesV2-GPT4-train.txt
```

输出文件：

```text
data/tokenizer/TinyStoriesV2-GPT4-train/TinyStoriesV2-GPT4-train.npy
```

当前接口中 `--text_path` 和 `--save_path` 使用下划线。编码会先写入分片，再合并为一个 `.npy` 文件。确认程序完成、数据能解码回原始文本后再训练；不要仅凭文件存在或大小判断编码成功。

### 7.3 训练模型

```powershell
llmfs model-train `
  --preset mid `
  --src data/tokenizer/TinyStoriesV2-GPT4-train/TinyStoriesV2-GPT4-train.npy `
  --vocab-size 32000 `
  --device cuda
```

这个示例沿用 `mid` 的 6 层、隐藏维度 384、上下文长度 256、batch size 8 和 10,000 次迭代，将词表大小覆盖为 32,000。10,000 次迭代不等于完整遍历语料 10,000 遍；每次迭代随机抽取一个 batch。

检查点保存在 `data/checkpoints/TinyStoriesV2-GPT4-train/`。最终权重保存在：

```text
data/model_params/mid_TinyStoriesV2-GPT4-train.pt
```

当前训练代码会在 `--out` 的文件名中追加语料文件名。例如 `--out data/model_params/mid.pt` 最终得到上面的路径。

### 7.4 加载权重并生成文本

```powershell
llmfs generate `
  --preset mid `
  --params-path data/model_params/mid_TinyStoriesV2-GPT4-train.pt `
  --vocab-file data/bpe_train/TinyStoriesV2-GPT4-train/vocab.pkl `
  --merges-file data/bpe_train/TinyStoriesV2-GPT4-train/merges.pkl `
  --vocab-size 32000 `
  --device cuda `
  --prompt "Once upon a time" `
  --max-new-tokens 100 `
  --temperature 0.8 `
  --top-p 0.9
```

生成时必须使用训练对应的词表和模型结构。训练覆盖过 `--vocab-size`、`--num-layers`、`--d-model`、`--d-ff`、`--num-heads`、`--context-length` 或 `--rope-theta` 时，生成也要提供相同的值；preset 不会自动读取训练时的覆盖参数。

当前 preset 中的数据和模型默认路径不对应以上生成目录，因此示例显式指定这些路径。生成结果包含输入提示词，目前没有遇到结束 token 自动停止的逻辑。

### 7.5 查看帮助

```powershell
llmfs --help
llmfs bpe-train --help
llmfs tokenize --help
llmfs tokenize encode --help
llmfs tokenize decode --help
llmfs tokenize train_mode --help
llmfs model-train --help
llmfs generate --help
```

特殊 token 默认包含 `<|endoftext|>`，`--special-token` 会在默认列表后追加，训练 BPE、编码和生成时应使用一致的特殊 token 设置。

## 8. 测试

在已安装项目的同一个 Python 环境中运行，无需额外安装测试包：

```powershell
python -m unittest discover -s tests -v
```

测试覆盖命令行参数到 dataclass 的传递、preset 解析和覆盖，以及 Tokenizer 编码、解码。接口测试通过 mock 避免实际启动训练或生成，不代表已完成模型质量评估。

如果安装了开发依赖，也可以运行 `python -m pytest -q`。

提交源码时保留 `src/`、`tests/`、`.gitignore`、`pyproject.toml`、`README.md` 和 `data/` 中的 `.gitkeep`。`data/` 下除 `.gitkeep` 外的文件、Python 缓存、日志和 `.vscode/` 已配置忽略；已经被 Git 跟踪的文件仍需单独检查。
## 9. 当前局限

目前项目主要用于学习和验证语言模型的基础原理，距离完整、高效的语言模型训练系统仍有较大差距。

具体局限：
- Tokenizer的encode优化还能进一步，现有encode功能效率低
- 模型训练阶段的分段训练还未完善，但是保留了load_checkpoint接口
- 数据集加载和训练效率都有待提高
- 在tokenize的时候只能用2000万token为一次文件，可能导致性能下降
- 不推荐large preset

## 10. 学习资料

- [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

**这个项目是一次从基础原理出发，逐步理解和实现语言模型的实践记录。**

代码仍在持续完善中。

## 数据集准备

本项目使用微软研究团队发布的 **TinyStoriesV2-GPT4** 数据集进行语言模型训练。

数据集下载地址：

| 文件 | 下载链接 |
|---|---|
| 训练集 | [TinyStoriesV2-GPT4-train.txt](https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt) |
| 验证集 | [TinyStoriesV2-GPT4-valid.txt](https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt) |

**数据集主页：** [TinyStories - Hugging Face](https://huggingface.co/datasets/roneneldan/TinyStories)

下载完成后，将数据文件放入项目的 `data/corpus/` 目录。

后续通过项目提供的 Tokenizer 训练与数据预处理命令，生成对应的词表、合并规则及模型训练数据。

**注意：** 数据集、Tokenizer 训练产物、预处理数据及模型权重均不纳入 Git 版本管理，需要用户自行下载或生成。
