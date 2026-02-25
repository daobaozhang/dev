# 🐳 Docker 部署指南 (适用于绿联 NAS / 群晖 / 本地)

本项目已经支持 Docker 一键部署，无需在宿主机安装 Python 环境。

## 🚀 方式一：Docker Compose 一键部署 (推荐)

如果您在 NAS 上使用 **Container Manager** 或支持 `docker-compose` 的环境：

1.  将整个项目文件夹上传到您的 NAS 目录 (例如 `/docker/free-llm-scanner`)。
2.  确保文件夹内包含 `docker-compose.yml` 和 `Dockerfile`。
    
    **⚠️ 关键检查点：您的文件夹结构必须长这样：**
    ```text
    /docker/freellm/
    ├── docker-compose.yml   (或者 .yaml)
    ├── Dockerfile          (注意：没有任何后缀名！不要叫 Dockerfile.txt)
    ├── requirements.txt
    ├── web_app.py
    └── ...其他源码文件
    ```
    
    > **常见错误**：
    > *   Windows 上上传时，`Dockerfile` 变成了 `Dockerfile.txt` (请开启显示文件扩展名检查)。
    > *   `Dockerfile` 文件名拼写错误（比如首字母没大写）。
    > *   只上传了 docker-compose.yml 而忘记上传其他文件。

3.  通过 SSH 进入目录，运行：
    ```bash
    docker-compose up -d --build
    ```
4.  或者在 NAS 的 Docker 管理界面中，选择"项目" -> "新增"，选择该路径下的 `docker-compose.yml` 启动。
5.  访问：`http://[NAS_IP]:8003`

## 🛠️ 方式二：手动构建镜像

如果您想手动构建镜像：

1.  **构建镜像**：
    ```bash
    docker build -t free-llm-scanner .
    ```

2.  **运行容器**：
    ```bash
    docker run -d \
      --name free-llm-scanner \
      --restart always \
      -p 8003:8000 \
      -e TZ=Asia/Shanghai \
      free-llm-scanner
    ```

## 📝 注意事项

*   **端口映射**：默认将容器内部的 `8000` 端口映射到了宿主机的 `8003` (防止与常用服务冲突)。您可以修改 `docker-compose.yml` 中的 `- "8003:8000"` 来更改端口。
*   **Key 存储**：本项目的 API Key 存储在您浏览器的 `localStorage` 中，**不会** 存储在服务器或 Docker 容器内。因此，重启容器、升级镜像都不会丢失您的 Key。换浏览器/电脑访问则需要重新输入。
*   **网络问题**：如果您在 NAS 上无法拉取 Python 镜像或访问 Github，请配置国内 Docker 镜像加速。

祝您使用愉快！✨
