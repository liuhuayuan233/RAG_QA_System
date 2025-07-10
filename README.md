# RAG知识问答系统

基于检索增强生成（RAG）技术的智能知识问答系统，专注于医疗健康领域的垂直应用。

## 🌟 项目特色

- 🤖 **智能问答**: 基于DeepSeek-R1模型的专业医疗知识问答
- 🔍 **语义检索**: BGE-large-zh-v1.5中文嵌入模型，精准理解中文医疗术语
- 📚 **多格式支持**: PDF、Word、TXT、Markdown、JSONL等文档格式
- 🎯 **医疗专业**: 专门针对医疗健康领域优化的RAG流程
- 📍 **答案溯源**: 清晰标注回答来源，确保可信度
- 💻 **友好界面**: Streamlit构建的现代化Web界面

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装Python依赖
pip install streamlit langchain langchain-community chromadb
pip install sentence-transformers huggingface-hub
pip install openai python-dotenv
pip install PyMuPDF python-docx pandas numpy

# 配置API密钥
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### 2. 准备数据

将文档放入 `documents/` 目录：
```
documents/
├── 医疗问答.jsonl
├── 健康知识.pdf
├── 医学文档.docx
└── ...
```

### 3. 构建知识库

```bash
# 构建向量数据库
python src/build_database.py
```

### 4. 启动系统

```bash
# 运行Web应用
streamlit run app.py
```

访问 `http://localhost:8501` 开始使用！

## 📁 项目结构

```
RAG_QA_System/
├── app.py                      # Streamlit主应用
├── .env                        # 环境变量配置
├── config/
│   └── config.py              # 系统配置
├── src/
│   ├── build_database.py      # 向量库构建
│   ├── document_processor.py  # 文档处理
│   ├── vector_store.py        # 向量存储
│   ├── retriever.py           # 检索器
│   ├── qa_chain.py            # 问答链
│   └── utils.py               # 工具函数
├── documents/                  # 文档目录
├── chroma_db/                 # 向量数据库
└── README.md                  # 项目说明
```

## ⚙️ 核心技术

- **语言模型**: DeepSeek-R1 (通过OpenAI API兼容接口)
- **嵌入模型**: BAAI/bge-large-zh-v1.5 (中文优化)
- **向量数据库**: ChromaDB (本地持久化)
- **Web框架**: Streamlit (现代化界面)
- **文档处理**: PyMuPDF, python-docx (多格式支持)

## 🔧 配置说明

### 环境变量 (.env)

```env
# API配置
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1

# 文档处理
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_DOCUMENT_SIZE=10000000

# 检索配置
TOP_K_RETRIEVAL=5
SIMILARITY_THRESHOLD=0.7

# 数据库路径
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

### 系统配置 (config/config.py)

主要配置项包括：
- 嵌入模型选择
- 文档分块策略
- 检索参数调优
- 医疗专业提示词

## 💡 使用指南

### 文档上传
1. 将文档放入 `documents/` 目录
2. 支持格式：PDF、DOCX、TXT、MD、JSONL
3. 运行构建脚本：`python src/build_database.py`

### 智能问答
1. 在Web界面输入医疗健康相关问题
2. 系统自动检索相关文档
3. 生成专业、准确的回答
4. 显示答案来源和相关度评分

### 答案溯源
- 每个回答都标注参考文档来源
- 显示相关度评分
- 支持查看原文片段
- 确保回答的可信度和专业性

## 🎯 医疗领域特色

### 专业提示词
系统内置医疗专业提示词，确保：
- 回答准确、专业、有针对性
- 明确标注医疗免责声明
- 建议严重症状及时就医
- 强调不能替代专业医疗诊断

### 安全提醒
- 所有医疗回答附带免责声明
- 强调仅供健康教育和参考
- 提醒用户及时就医咨询专业医生

## 🔬 技术亮点

### 1. 先进的中文理解
- BGE-large-zh-v1.5专为中文优化
- 精准理解医疗术语和专业表达
- 支持复杂医疗场景的语义检索

### 2. 智能文档处理
- 自动识别文档格式
- 智能文本分块和清洗
- 医疗文档结构化处理

### 3. 高效向量检索
- ChromaDB高性能向量数据库
- 多种相似度计算方法
- 可配置的检索策略

### 4. 专业问答生成
- 上下文感知的回答生成
- 医疗专业性验证
- 多轮对话支持

## 📈 性能特点

- **处理能力**: 支持大规模文档处理（单文档最大10MB）
- **响应速度**: 平均检索响应时间 < 1秒
- **准确性**: 基于专业医疗数据的高质量回答
- **扩展性**: 模块化设计，易于功能扩展

## 🛠️ 开发说明

### 模块说明
- `DocumentProcessor`: 文档解析和预处理
- `VectorStore`: 向量化存储和管理
- `Retriever`: 智能检索和重排序
- `QAChain`: 问答链和LLM集成
- `Config`: 统一配置管理

### 扩展开发
- 支持新的文档格式处理
- 集成其他嵌入模型
- 添加更多LLM支持
- 自定义检索策略

## ⚠️ 重要说明

**医疗免责声明**: 本系统提供的医疗健康信息仅供教育和参考用途，不能替代专业医疗诊断、治疗或建议。如有健康问题或疑虑，请及时咨询专业医生。


**RAG知识问答系统** - 让AI理解医疗，让知识触手可及 🏥✨
