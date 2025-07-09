# 安装和运行指南

## 环境要求

- Python 3.8 或更高版本
- Windows 10/11 (本指南针对Windows环境)
- 至少 4GB 内存
- 网络连接（用于下载模型）

## 安装步骤

### 1. 创建虚拟环境

```powershell
# 创建虚拟环境
python -m venv rag_env

# 激活虚拟环境
rag_env\Scripts\activate
```

### 2. 安装依赖包

```powershell
# 升级pip
python -m pip install --upgrade pip

# 安装依赖包
pip install -r requirements.txt
```

### 3. 配置环境变量

```powershell
# 复制环境变量配置文件
copy .env.example .env

# 编辑 .env 文件，填入你的API密钥
notepad .env
```

在 `.env` 文件中设置：

```env
# OpenAI API配置
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1

# 向量数据库配置
CHROMA_PERSIST_DIRECTORY=./chroma_db
VECTOR_DIMENSION=1024

# 系统配置
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=5
```

### 4. 准备文档

将你的文档放入 `documents` 目录：

```powershell
# 创建文档目录（如果不存在）
mkdir documents

# 将文档复制到目录中
# 支持的格式：PDF, DOCX, TXT, MD
```

### 5. 构建向量知识库

```powershell
# 运行向量库构建脚本
python scripts/build_vector_store.py

# 或者使用交互模式
python scripts/build_vector_store.py --interactive
```

### 6. 启动Web应用

```powershell
# 启动Streamlit应用
streamlit run app.py
```

## 常见问题解决

### 1. 依赖包安装失败

**问题**: 某些包安装失败

**解决方案**:
```powershell
# 更新pip和setuptools
python -m pip install --upgrade pip setuptools

# 分别安装可能有问题的包
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
pip install chromadb
```

### 2. 模型下载失败

**问题**: BGE模型下载失败

**解决方案**:
```powershell
# 手动下载模型
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-zh-v1.5')"
```

### 3. 向量库构建失败

**问题**: 向量库构建过程中出错

**解决方案**:
```powershell
# 检查文档格式是否支持
python -c "from config.config import Config; print(Config.SUPPORTED_EXTENSIONS)"

# 清理向量库重新构建
rmdir /s chroma_db
python scripts/build_vector_store.py --rebuild
```

### 4. OpenAI API问题

**问题**: API调用失败

**解决方案**:
- 检查API密钥是否正确
- 确认网络连接正常
- 检查API额度是否充足

### 5. 内存不足

**问题**: 运行时内存不足

**解决方案**:
- 减少批处理大小
- 使用更小的嵌入模型
- 增加系统内存

## 测试系统

### 1. 运行测试脚本

```powershell
# 运行完整测试
python tests/test_system.py

# 交互式测试
python tests/test_system.py interactive
```

### 2. 验证功能

测试以下功能：
- 文档处理
- 向量存储
- 文档检索
- 问答生成
- 答案溯源

## 高级配置

### 1. 使用其他LLM

编辑 `config/config.py`：

```python
# 使用其他API
OPENAI_API_BASE = "https://api.deepseek.com/v1"  # DeepSeek API
OPENAI_API_BASE = "https://api.zhipuai.cn/v1"    # 智谱API
```

### 2. 调整模型参数

```python
# 调整块大小
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# 调整检索数量
TOP_K_RETRIEVAL = 3

# 调整相似度阈值
SIMILARITY_THRESHOLD = 0.6
```

### 3. 使用GPU加速

```python
# 在vector_store.py中修改
model_kwargs={'device': 'cuda'}  # 使用GPU
```

## 性能优化

### 1. 文档预处理

```powershell
# 清理文档格式
# 移除不必要的空白和特殊字符
# 统一文档编码为UTF-8
```

### 2. 向量库优化

```powershell
# 定期重建向量库
python scripts/build_vector_store.py --rebuild

# 使用更高效的嵌入模型
# 可以尝试：sentence-transformers/all-MiniLM-L6-v2
```

### 3. 系统监控

```powershell
# 监控系统资源使用
# 查看内存和CPU使用情况
# 监控API调用次数
```

## 部署建议

### 1. 本地部署

```powershell
# 使用更稳定的WSGI服务器
pip install gunicorn
gunicorn --bind 0.0.0.0:8000 app:app
```

### 2. Docker部署

```dockerfile
# 创建Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 3. 云部署

考虑使用：
- Streamlit Cloud
- Azure Container Instances
- AWS ECS
- Google Cloud Run

## 维护建议

### 1. 定期更新

```powershell
# 更新依赖包
pip install --upgrade -r requirements.txt

# 更新模型
# 检查是否有新版本的BGE模型
```

### 2. 数据备份

```powershell
# 备份向量库
xcopy /s chroma_db backup_chroma_db\

# 备份文档
xcopy /s documents backup_documents\
```

### 3. 日志监控

```powershell
# 查看系统日志
type rag_system.log

# 监控错误日志
findstr "ERROR" rag_system.log
```

## 扩展功能

### 1. 添加新的文档格式

编辑 `src/document_processor.py`：

```python
# 添加新的文档处理方法
def _extract_excel_text(self, file_path: str) -> str:
    # 实现Excel文档处理
    pass
```

### 2. 集成新的嵌入模型

编辑 `src/vector_store.py`：

```python
# 使用不同的嵌入模型
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

### 3. 添加更多LLM支持

编辑 `src/qa_chain.py`：

```python
# 添加对其他LLM的支持
from langchain.llms import Ollama
# 使用本地LLM
```

---

按照以上步骤，你应该能够成功安装和运行RAG知识问答系统。如果遇到问题，请检查日志文件或联系技术支持。
