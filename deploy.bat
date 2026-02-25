@echo off
chcp 65001 >nul
echo ==========================================
echo   Free LLM Scanner 一键部署脚本
echo   使用 docker-compose.yml 部署
echo ==========================================
echo.

REM 检查Docker是否安装
docker version >nul 2>&1
if errorlevel 1 (
    echo [错误] Docker 未安装或未启动
    pause
    exit /b 1
)

echo 部署选项:
echo   1) 构建镜像并启动 (代码有大改动时用)
echo   2) 仅启动容器 (代码小改动后用)
echo   3) 停止容器
echo   4) 查看日志
echo   5) 查看状态
echo   0) 退出
echo.

set /p choice=请输入选项 (0-5): 

if "%choice%"=="1" (
    echo.
    echo [1] 构建镜像并启动...
    docker-compose up -d --build
    echo.
    echo ==========================================
    echo   部署完成！
    echo   访问地址: http://你的NASIP:8003
    echo ==========================================
)

if "%choice%"=="2" (
    echo.
    echo [2] 仅启动容器...
    docker-compose up -d
    echo.
    echo 完成！
)

if "%choice%"=="3" (
    echo.
    echo [3] 停止容器...
    docker-compose down
    echo.
    echo 已停止！
)

if "%choice%"=="4" (
    echo.
    echo [4] 查看日志 (按Ctrl+C退出)...
    docker-compose logs -f
)

if "%choice%"=="5" (
    echo.
    echo [5] 容器状态:
    docker-compose ps
)

if "%choice%"=="0" (
    exit /b 0
)

pause