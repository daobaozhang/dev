# 使用官方轻量级 Python 镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 防止 Python 生成 .pyc 文件，并启用无缓冲输出 (便于查看 Docker 日志)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 复制依赖文件并安装 (利用 Docker 缓存层)
COPY requirements.txt .
# 使用清华源加速依赖安装
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目源代码
COPY . .

# 验证必要目录存在
RUN ls -la static templates src

# 暴露 8000 端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "web_app:app", "--host", "0.0.0.0", "--port", "8000"]
