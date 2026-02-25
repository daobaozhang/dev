#!/bin/bash

# ==========================================
# Free LLM Scanner 一键部署脚本
# 使用 docker-compose.yml 部署
# ==========================================

echo "=========================================="
echo "  Free LLM Scanner 部署脚本"
echo "=========================================="

echo ""
echo "部署选项:"
echo "  1) 构建镜像并启动 (代码有大改动时用)"
echo "  2) 仅启动容器 (代码小改动后用)"
echo "  3) 停止容器"
echo "  4) 查看日志"
echo "  5) 查看状态"
echo "  0) 退出"
echo ""

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

read -p "请输入选项 (0-5): " choice

case $choice in
    1)
        echo ""
        echo "[1] 构建镜像并启动..."
        docker-compose up -d --build
        
        echo ""
        echo "=========================================="
        echo "  部署完成！"
        echo "  访问地址: http://你的NASIP:8003"
        echo "=========================================="
        ;;
    2)
        echo ""
        echo "[2] 仅启动容器..."
        docker-compose up -d
        echo ""
        echo "完成！"
        ;;
    3)
        echo ""
        echo "[3] 停止容器..."
        docker-compose down
        echo ""
        echo "已停止！"
        ;;
    4)
        echo ""
        echo "[4] 查看日志 (按Ctrl+C退出)..."
        docker-compose logs -f
        ;;
    5)
        echo ""
        echo "[5] 容器状态:"
        docker-compose ps
        ;;
    0)
        exit 0
        ;;
esac
