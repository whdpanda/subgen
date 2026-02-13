# Docker (CPU) Smoke Test

> 可选：给团队 / 未来自己复现。  
> 如果你不想加 `docs` 文件，也可以不加；但这是“先跑通”的最低摩擦。

## 1) Build & run

```bash
docker compose up --build
```

## 2) Put a video under `./out`

Example:

- host: `./out/universe.webm`
- container: `/data/universe.webm`

## 3) Health

```bash
curl http://127.0.0.1:8000/health
```

## 4) Generate

```bash
curl -s -X POST http://127.0.0.1:8000/v1/subtitles/generate \
  -H "Content-Type: application/json" \
  -d '{"video_path":"/data/universe.webm","out_dir":"/data/out_api","max_passes":3}'
```

## 5) Check outputs

Host path: `./out/out_api/`

- `debug.log`
- `*.srt`
- report files (if produced)

## 6) Burn (optional)

```bash
curl -s -X POST http://127.0.0.1:8000/v1/subtitles/burn \
  -H "Content-Type: application/json" \
  -d '{"video_path":"/data/universe.webm","srt_path":"/data/out_api/xxx.zh.srt","out_dir":"/data/out_burn"}'
```
