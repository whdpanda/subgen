# 项目结构优化建议（面向可读性与可维护性）

本文基于当前仓库结构给出“可立即做 / 中期可做 / 长期可做”的优化建议，目标是：

- 降低新同学理解成本
- 降低模块耦合
- 提升测试可定位性
- 为后续扩展（多引擎、多语言、多工作流）预留空间

---

## 1. 已执行的低风险清理

- 删除仓库根目录的 `result.json`（本地运行产物，且未被代码引用）。
- 在 `.gitignore` 中加入 `result.json`，避免后续误提交本地结果快照。

---

## 2. 可以删除/移动的文件建议

### 2.1 根目录文档归并到 `docs/`

当前根目录有 `项目介绍.md`，建议迁移到：

- `docs/overview.md`（对外介绍）
- 或 `docs/architecture/overview.md`（偏工程化说明）

这样根目录可以保持“入口文件最少化”（`pyproject.toml`、`src/`、`tests/`、`docs/`）。

### 2.2 运行时产物统一落地到 `out/`

项目已忽略 `out/`，建议把所有示例输出（如 `.srt`、质量报告 JSON）统一放在 `out/`，避免根目录污染。

---

## 3. 可以合并的模块建议（控制碎片化）

> 原则：小模块不是问题，但“过细 + 横向跳转多”会增加认知负担。

### 3.1 `core/postprocess/` 可做职责分组

当前 `coalesce.py`、`dedupe.py`、`punct_split.py`、`zh_layout.py` 是“处理链上的多个算子”。
建议新增一个编排入口，例如：

- `core/postprocess/pipeline.py`（仅负责调用顺序）

保留现有文件作为原子算子，减少调用方直接拼装多个步骤。

### 3.2 `agent/tools/` 中 schema 与 envelope 可就近归并

如果 `schemas.py` 与 `envelope.py` 主要服务于工具调用协议，可考虑：

- 合并为 `tool_protocol.py`
- 或放入 `agent/tools/_internal/`，减少对外可见面

目的是让“工具能力文件”和“工具协议文件”分层更清晰。

---

## 4. 建议拆分的模块（降低单文件复杂度）

### 4.1 `core/pipeline.py` 建议按阶段拆分

可按阶段拆为：

- `pipeline/asr_stage.py`
- `pipeline/translate_stage.py`
- `pipeline/refine_stage.py`
- `pipeline/export_stage.py`
- `pipeline/orchestrator.py`

这样未来增加新 stage（例如术语增强、时间轴修复）时，不会持续膨胀一个中心文件。

### 4.2 `agent/loop.py` 与 `agent/runtime.py` 的边界再明确

建议分成“控制流”与“依赖装配”两层：

- loop 只处理状态推进、重试、终止条件
- runtime 只处理模型、工具、知识库注入

若已有此趋势，可通过命名（`*_orchestrator.py` / `*_bootstrap.py`）进一步强化。

---

## 5. 目录调整建议（面向扩展）

### 5.1 建议引入 `application` 与 `domain` 的轻分层

在不做大重构前提下，可渐进式整理为：

- `subgen/domain/`：纯数据结构与规则（字幕模型、质量规则）
- `subgen/application/`：工作流编排（pipeline、agent loop）
- `subgen/infrastructure/`：外部依赖实现（OpenAI、Whisper、文件 IO）

当前代码已具备雏形（`core/*` + `agent/*`），只需逐步迁移命名与目录。

### 5.2 测试目录镜像源码结构

建议从“按主题混合”逐步过渡到“按包镜像”：

- `tests/core/...`
- `tests/agent/...`
- `tests/integration/...`

并把 fixture 按测试域归档，例如：

- `tests/fixtures/subtitle_quality/...`

---

## 6. 推荐执行顺序（两周内可完成）

1. **第 1 步（低风险）**：清理产物文件、统一文档位置、补充目录说明。
2. **第 2 步（中风险）**：增加 `postprocess/pipeline.py`，减少调用方拼装细节。
3. **第 3 步（中风险）**：拆 `core/pipeline.py` 为 stage + orchestrator。
4. **第 4 步（中风险）**：重整 tests 目录映射。
5. **第 5 步（可选）**：逐步引入 domain/application/infrastructure 命名。

---

## 7. 何时“不建议”马上动结构

如果近期目标是快速验证算法效果（而非多人协作），可先不做大规模目录迁移，优先做：

- 模块 docstring 标准化
- 关键入口文件补充架构注释
- 增加端到端测试

以上投入小、收益快，也能为后续重构打基础。
