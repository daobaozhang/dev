# Docker 部署与更新指南

## 1. 首次部署 (若尚未运行)

如果您是第一次部署，请直接运行以下命令：

```bash
docker-compose up -d --build
```

这将构建镜像并启动容器。

## 2. 更新部署 (已运行中的容器)

如果您已经部署了旧版本，需要更新以包含"访客统计"功能，请执行以下步骤：
请在 NAS 的终端或者 SSH 中进入到项目目录 `CD free-llm-api-resources`

### 步骤 A: 停止旧容器
```bash
docker-compose down
```

### 步骤 B: 重新构建并启动
由于代码有修改 (添加了 SQLite 和新依赖)，我们需要强制通过 `--build` 参数重新构建镜像：

```bash
docker-compose up -d --build
```
> **注意**: 如果不加 `--build`，Docker 可能只会重启容器而不会更新代码。

### 步骤 C: 验证数据持久化
启动后，会在当前目录下自动生成 `data/visitor_stats.db` 文件。
- 请确保 `data` 目录有写入权限。
- 该目录已挂载到容器内，因此即使重启容器，统计数据也不会丢失。

## 3. 常见问题

**Q: 为什么统计数据没有变化？**
A: 统计逻辑有防刷机制，只有访问根路径 `/` 且状态码为 200 时才会记录。刷新页面应该会增加"总访问量"。

**Q: 权限问题？**
A: 如果遇到 `Permission denied` 错误，请尝试给 data 目录赋予权限: `chmod -R 777 data` (Linux/Mac) 或在 Windows 属性中检查权限。
