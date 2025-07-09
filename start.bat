@echo off
chcp 65001 >nul
echo ================================================
echo RAG 知识问答系统 - 快速启动脚本
echo ================================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] Python未安装或未添加到PATH
    echo 请先安装Python 3.8或更高版本
    pause
    exit /b 1
)

echo [1/6] 检查Python环境...
python --version

REM 创建虚拟环境
echo.
echo [2/6] 创建虚拟环境...
if not exist "rag_env" (
    python -m venv rag_env
    echo 虚拟环境创建成功
) else (
    echo 虚拟环境已存在
)

REM 激活虚拟环境
echo.
echo [3/6] 激活虚拟环境...
call rag_env\Scripts\activate.bat

REM 升级pip
echo.
echo [4/6] 升级pip...
python -m pip install --upgrade pip

REM 安装依赖
echo.
echo [5/6] 安装依赖包...
pip install -r requirements.txt

REM 检查环境变量文件
echo.
echo [6/6] 检查配置文件...
if not exist ".env" (
    echo 创建环境变量文件...
    copy .env.example .env
    echo.
    echo [重要] 请编辑 .env 文件，设置您的OpenAI API密钥
    echo 然后重新运行此脚本
    pause
    exit /b 1
)

REM 检查文档目录
if not exist "documents" (
    mkdir documents
    echo 已创建文档目录
)

echo.
echo ================================================
echo 安装完成！
echo ================================================
echo.
echo 接下来的步骤：
echo 1. 将文档放入 documents 目录
echo 2. 运行: python scripts/build_vector_store.py
echo 3. 运行: streamlit run app.py
echo.
echo 或者运行测试：
echo python tests/test_system.py
echo.

REM 询问是否继续
set /p choice="是否现在构建向量库？(y/N): "
if /i "%choice%"=="y" (
    echo.
    echo 构建向量库...
    python scripts/build_vector_store.py
    
    if %errorlevel% equ 0 (
        echo.
        echo 向量库构建成功！
        echo.
        set /p choice2="是否启动Web应用？(y/N): "
        if /i "%choice2%"=="y" (
            echo.
            echo 启动Web应用...
            streamlit run app.py
        )
    ) else (
        echo.
        echo 向量库构建失败，请检查错误信息
    )
)

pause
