# SubGen：高精度本地化字幕生成工具

> **High-Accuracy First Subtitle Generator (local-first)**

SubGen 是一个面向低资源语言和高准确度场景设计的字幕生成流水线，提供从音频提取、自动语音识别（ASR）、分段、翻译到字幕导出的完整流程。项目提供简单易用的命令行界面，通过本地 Whisper 模型（`faster-whisper`）实现高质量转写，并支持灵活的语义分割、词汇表应用和多种翻译器选项。对于中文输出，还可利用大语言模型进行润色。

## 功能特性

- **端到端流水线**：从视频/音频中提取音轨，经过 ASR、分段、翻译和修复后生成 SRT 文件，可选择单语或双语输出。
- **高精度 ASR**：采用 [Faster Whisper](https://github.com/guillaumekln/faster-whisper) 等本地模型，支持配置模型大小（如 `large-v3`）、计算设备（CPU/GPU）、精度 (`float16`/`int8`) 以及 beam size、best-of 和 VAD 过滤等参数。
- **音频预处理**：可选择无预处理、简单语音滤波或 [Demucs](https://github.com/facebookresearch/demucs) 分离，提高嘈杂环境下的识别质量。
- **灵活分段**：内置规则分段器和 OpenAI 语义分割器，可以设置软/硬最大片段长度、疑似尾听时长和每秒字符数阈值等，避免长句或短句影响字幕可读性。
- **修复与启发式**：通过尾听重试、句末标点切分和极短片段合并等策略修复分段结果。`refine/risk.py` 中的启发式规则会根据置信度、数字过多、术语重复或机器翻译痕迹重新识别“风险片段”，确保结果稳定可靠。
- **多翻译器支持**：默认使用 [NLLB 200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M) 进行本地翻译，也可选择 OpenAI 模型。`auto_non_en` 模式会根据源语言自动判断是否需要翻译。支持设置目标语言、模型名称和运行设备。
- **术语表应用**：可通过 `--glossary` 传入 JSON 格式的术语映射，翻译完成后自动替换指定词汇，保证专业名词一致性。
- **中文润色**：提供基于 OpenAI 的文字重写功能，用于改善中文句子的自然度。该功能使用系统提示保证译文忠实，不解释或删除信息。
- **缓存机制**：ASR 结果会根据音频内容及模型参数生成缓存键，重复运行时可直接命中缓存加快速度。用户可通过 `--no-use-cache` 强制重新转写。
- **可扩展的 CLI**：基于 [Typer](https://typer.tiangolo.com/) 构建的命令行接口，支持自动生成帮助文档，可自由组合各类参数以适配不同场景。

## 安装

环境要求：Python 3.10 及以上，建议使用虚拟环境；如果使用 GPU，需要正确安装 CUDA 驱动。

```bash
# 推荐从 Git 安装最新版（需要 git）
pip install git+https://github.com/whdpanda/subgen.git

# 或者克隆仓库本地安装
git clone https://github.com/whdpanda/subgen.git
cd subgen
pip install -e .

# 依赖：ffmpeg 用于音频提取（需预先安装在系统中）
```

如需使用 OpenAI 功能（分段/翻译/润色），请将 API 密钥放入环境变量：

```bash
export OPENAI_API_KEY=sk-xxxx
```

## 快速上手

以下命令将英文视频 `video.mp4` 转写为中英双语字幕，并保存到 `./out` 目录：

```bash
subgen gen \
  --input video.mp4 \
  --out ./out \
  --lang en \
  --to zh \
  --preprocess speech_filter \
  --asr_model large-v3 \
  --asr_device cuda \
  --segmenter openai \
  --openai_segment_model gpt-5-mini \
  --translator_name auto_non_en \
  --translator_model facebook/nllb-200-distilled-600M \
  --emit bilingual-only
```

命令解释：

1. 提取并过滤音频（`speech_filter`）后，使用本地 `large-v3` Whisper 模型在 GPU 上转写。
2. 使用 OpenAI 语义分割（模型 `gpt-5-mini`）生成字幕片段。
3. 根据源语言自动决定翻译器：非中文情况下调用 NLLB；若需要远程翻译可改为 `--translator_name openai`。
4. 输出中文译文（`--to zh`）并生成双语字幕文件（`--emit bilingual-only`）。
5. 所有参数均可通过 `subgen gen --help` 查看详细说明。

### 常用参数

| 参数                              | 说明                                                     |
| ------------------------------- | ------------------------------------------------------ |
| `--preprocess`                  | 音频预处理方式：`none`、`speech_filter`、`demucs`                |
| `--asr_model`                   | Whisper 模型名称，如 `base`、`small`、`large-v3`               |
| `--asr_device`                  | ASR 运行设备：`cpu`、`cuda`、`auto`                           |
| `--segmenter`                   | 分段器类型：`rule` 或 `openai`                                |
| `--soft_max`/`--hard_max`       | 分段软/硬最长时长（秒）                                           |
| `--suspect_dur`/`--suspect_cps` | 尾听疑似片段长度和每秒字符阈值                                        |
| `--translator_name`             | 翻译器：`auto_non_en`、`nllb` 或 `openai`                    |
| `--openai_translate_model`      | 当使用 OpenAI 翻译时所用模型                                     |
| `--glossary`                    | 术语表 JSON 文件路径                                          |
| `--emit`                        | 输出类型：`all`（单语译文+双语）、`literal`（仅源文本）、`bilingual-only` 等 |

## 术语表格式

术语表文件需为 JSON 格式，键为源语言短语，值为目标语言短语。例如：

```json
{
  "LLM": "大语言模型",
  "pipeline": "流水线",
  "ASR": "自动语音识别"
}
```

指定后将自动应用替换于译文。

## 路线图

* 支持长视频的并行处理与流式解码
* 集成更多翻译器和分段模型
* 提供性能评测脚本和单元测试

## 许可证

仓库尚未指定开源许可证。若需用于商业场景，请先联系作者或按照自述协议使用。