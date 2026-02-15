# SubGen Docs 导航

## 快速入口
- **部署入口（Deployment Entry）**
  - Docker 本地冒烟：[`docker_smoke.md`](./docker_smoke.md)
  - Kubernetes 部署与运维：[`DEPLOY_K8S.md`](./DEPLOY_K8S.md)
- **异步 Jobs API**：[`API_ASYNC_JOBS.md`](./API_ASYNC_JOBS.md)
- **运维手册（日志/指标/常见故障）**：[`OPERATIONS.md`](./OPERATIONS.md)

## Kubernetes Quickstart
1. 阅读 [`DEPLOY_K8S.md`](./DEPLOY_K8S.md) 的前置条件。
2. 在仓库根目录执行：`kubectl apply -k k8s/`。
3. 使用文档中的 `port-forward` 与健康检查命令验证服务。
4. 遇到排障需求时，查看 [`OPERATIONS.md`](./OPERATIONS.md)。

## 其他文档
- 项目介绍与架构整理：[`overview.md`](./overview.md)
- 项目结构优化说明：[`project_structure_optimization.md`](./project_structure_optimization.md)
- Agent 知识库文档入口：[`knowledge/README.md`](./knowledge/README.md)
