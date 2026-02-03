# Config

## Knowledge Base (RAG)
Environment variables:
- `SUBGEN_KB_DIR`: chroma persist directory (default: ./.subgen_kb)
- `SUBGEN_KB_COLLECTION`: collection name (default: subgen-kb)

Index script:
- `python scripts/build_kb.py --src docs/knowledge --persist ./.subgen_kb --collection subgen-kb`
