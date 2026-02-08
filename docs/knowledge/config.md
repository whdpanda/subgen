# Config

## Knowledge Base (RAG)
Environment variables:
- `SUBGEN_KB_DIR`: chroma persist directory (default: `./.subgen_kb`)
- `SUBGEN_KB_COLLECTION`: collection name (default: `subgen-kb`)

Index script:
- `python scripts/build_kb.py --src docs/knowledge --persist ./.subgen_kb --collection subgen-kb`

Rebuild from scratch (PR#4c):
- `python scripts/build_kb.py --src docs/knowledge --persist ./.subgen_kb --collection subgen-kb --reset`

## Agent runtime (PR#4c)
Environment variables:
- `SUBGEN_AGENT_MODEL`: LLM model for agent runtime (default: `gpt-5.2`)
- `SUBGEN_AGENT_MAX_STEPS`: max tool-call steps for a single run (default: `16`)
- `SUBGEN_QUALITY_MAX_PASSES`: max fix/recheck passes after generation (default: `3`)

Defaults (unless user overrides via tool args):
- `target_lang=zh`
- `emit=zh-only`
- `zh_layout=true`
