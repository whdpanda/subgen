# Docker (CPU) Smoke Test

## 1) 启动容器

```bash
docker compose up --build
```

## 2) 挂载目录规则（重要）

`docker-compose.yml` 已配置目录挂载：

- host: `./out`
- container: `/data`

例如你把视频放在宿主机 `./out/universe.webm`，容器内对应路径就是 `/data/universe.webm`。

> **强调：API 只接受容器内路径**（`/data/...`），不要传宿主机路径（如 `./out/...`）。

## 3) 健康检查

```bash
curl http://127.0.0.1:8000/health
```

## 4) API 调用示例

### 4.1 generate

```bash
curl -s -X POST http://127.0.0.1:8000/v1/subtitles/generate \
  -H "Content-Type: application/json" \
  -d '{"video_path":"/data/universe.webm","out_dir":"/data/out_api","max_passes":3}'
```

### 4.2 fix

```bash
curl -s -X POST http://127.0.0.1:8000/v1/subtitles/fix \
  -H "Content-Type: application/json" \
  -d '{"srt_path":"/data/out_api/universe.zh.srt","out_dir":"/data/out_fix"}'
```

### 4.3 burn

```bash
curl -s -X POST http://127.0.0.1:8000/v1/subtitles/burn \
  -H "Content-Type: application/json" \
  -d '{"video_path":"/data/universe.webm","srt_path":"/data/out_fix/universe.zh.fixed.srt","out_dir":"/data/out_burn"}'
```

## 5) 检查产物

宿主机路径 `./out/` 下可看到容器写出的产物：

- `./out/out_api/`
- `./out/out_fix/`
- `./out/out_burn/`
