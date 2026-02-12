# SubGen 项目介绍（重构整理版）

## 1. 项目定位
SubGen 是一个**高质量优先**的视频字幕生成项目，核心目标是：
- 从视频中提取音频并进行 ASR；
- 做切分、翻译与中文排版；
- 输出可审查、可修复、可烧录（burn-in）的字幕结果；
- 支持 Agent 闭环：`生成 -> 质检 -> 修复 -> 再质检 ->（可选）烧录`。

目前代码已经形成两层能力：
1. **Core Pipeline（src/subgen/core）**：偏业务算法和处理链。
2. **Agent Tools（src/subgen/agent/tools）**：偏工具协议、参数校验和错误回传。

---

## 2. 目录结构概览
- `src/subgen/core/`：主流水线、ASR、切分、翻译、后处理、导出、质量检查。
- `src/subgen/agent/`：Agent runtime + 闭环调度。
- `src/subgen/agent/tools/`：可被 LLM 调用的工具入口。
- `docs/knowledge/`：给 Agent 的知识文档（RAG 资料）。
- `tests/`：工具契约、闭环行为、基础 API 测试。

---

## 3. 本轮整理与优化
本次做了几项“低风险、可维护性收益高”的结构优化：

### 3.1 合并重复参数定义（减少散落与不一致）
- 在 `schemas.py` 中提取了 `QualityProfileOverrides`，让 `QualityToolArgs` 与 `FixToolArgs` 复用同一组质量阈值参数。
- 这样可避免未来调默认值时出现“一个改了另一个漏改”的问题。

### 3.2 合并重复的 Path 序列化逻辑
- 在 `tool_utils.py` 新增：
  - `path_to_str()`
  - `path_values_to_str()`
- 用于统一工具输出时把 `Path` 转 `str` 的行为。
- `run_subgen_pipeline_tool.py` 与 `burn_subtitles_tool.py` 已切换到公共工具函数，减少重复实现。

### 3.3 清理工具注册入口可读性
- 重写 `src/subgen/agent/tools/__init__.py`：
  - 去除异常字符（BOM）并统一导入风格；
  - 精简 typing 用法（现代 `list[...]` / `dict[...]`）；
  - 保留原有校验逻辑（工具名完整性、顺序一致性、重复检测）。

### 3.4 修复测试环境导入路径问题
- 在 `pyproject.toml` 的 pytest 配置中增加 `pythonpath = ["src"]`。
- 这样可直接在仓库根目录运行 `pytest`，避免 `ModuleNotFoundError: subgen`。

---

## 4. 代码维护建议（下一阶段）
以下建议基于当前代码结构，建议按优先级推进：

### P0（优先）
1. **统一 Tool 错误返回构建器**
   - 当前多个 tool 内各自维护 `_fail_flat` / `_ok_flat` 变体。
   - 建议在 `tool_utils.py` 再抽一层通用 builder（或 dataclass + helper），确保字段稳定。

2. **补齐 CLI 入口与打包一致性检查**
   - `pyproject.toml` 中声明了 `subgen = "subgen.cli.main:app"`，但当前源码树中未见 `subgen/cli/main.py`。
   - 建议补齐或调整 entrypoint，避免安装后命令不可用。

3. **无引用模块标记弃用策略**
   - 例如 `agent/tools/envelope.py` 看起来已不在主流程调用。
   - 建议先标记 `deprecated` 并加测试，再在后续版本移除，避免外部脚本潜在依赖被突然破坏。

### P1（重要）
4. **去重 runtime 与 loop 中的 envelope 兼容逻辑**
   - `runtime.py` 与 `loop.py` 都有 `_is_envelope` / flatten 逻辑，建议共用一个适配函数。

5. **提升配置对象可验证性**
   - `PipelineConfig` 目前为 dataclass，若要增强字段校验与错误提示，可评估迁移到 Pydantic model。

6. **质量报告文件命名策略统一**
   - 目前 quality/fix 工具各有命名规则；建议集中到 `quality/report_naming.py` 一处维护。

### P2（增强）
7. **增加分层单测覆盖**
   - 增加“纯函数级”测试（例如 path 解析、输出 payload 结构），减少 e2e 才能发现的问题。

8. **引入 ruff/mypy 的 CI 强制门禁**
   - 目前依赖已声明，但建议在 CI 固定执行，控制风格和类型回归。

---

## 5. 后续维护建议（实践）
- 每次新增工具时，先补“schema 契约测试”再写实现。
- 对于 Agent 可见字段（如 `ok/report_path/meta.error`）保持向后兼容。
- 严格区分：
  - `core` 负责业务真逻辑；
  - `agent/tools` 负责 I/O 契约、容错和字符串化。

---

## 6. 一句话总结
当前 SubGen 已具备可用的“生成 + 质检 + 修复 + 烧录”闭环能力；本轮重点把**重复定义、重复工具函数和测试可运行性**做了收敛，后续若继续推进统一错误协议与 CLI/CI 完整性，项目可维护性会显著提升。
