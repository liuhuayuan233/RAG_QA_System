# RAG知识问答系统

基于检索增强生成（RAG）技术的垂直领域知识问答系统，支持文档处理、向量库构建和智能问答。

## 功能特点

- 📚 **多格式文档处理**：支持PDF、Word、TXT等多种文档格式
- 🔍 **智能文档检索**：基于BGE中文嵌入模型的语义检索
- 💬 **智能问答**：集成大语言模型的问答生成
- 📍 **答案溯源**：标注回答的具体文档来源
- 🎯 **垂直领域优化**：针对特定领域数据优化的RAG流程

## 技术栈

- **嵌入模型**：BGE-large-zh-v1.5（中文优化）
- **向量数据库**：ChromaDB / FAISS
- **语言模型**：OpenAI GPT系列 / 支持其他LLM
- **框架**：LangChain
- **前端**：Streamlit
- **文档处理**：PyPDF、python-docx

## 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone <your-repo-url>
cd RAG_QA_System

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入API密钥
```

### 2. 数据准备

将文档放入 `documents/` 目录：

```
documents/
├── 行业报告.pdf
├── 技术文档.docx
├── 知识库.txt
└── ...
```

### 3. 构建向量库

```bash
python scripts/build_vector_store.py
```

### 4. 启动问答系统

```bash
streamlit run app.py
```

## 项目结构

```
RAG_QA_System/
├── app.py                  # Streamlit Web应用
├── requirements.txt        # 依赖包
├── .env.example           # 环境变量示例
├── config/
│   └── config.py          # 配置文件
├── src/
│   ├── document_processor.py  # 文档处理
│   ├── vector_store.py        # 向量库管理
│   ├── retriever.py          # 检索器
│   ├── qa_chain.py           # 问答链
│   └── utils.py              # 工具函数
├── scripts/
│   └── build_vector_store.py # 向量库构建脚本
├── documents/              # 文档目录
├── chroma_db/             # 向量数据库
└── tests/                 # 测试文件
```

## 使用指南

### 支持的文档格式

- PDF文件（.pdf）
- Word文档（.docx）
- 纯文本文件（.txt）
- Markdown文件（.md）

### 核心功能

1. **文档上传与处理**
   - 自动识别文档格式
   - 智能文本分块
   - 向量化存储

2. **智能检索**
   - 基于BGE模型的语义检索
   - 支持多种相似度计算
   - 可配置检索数量

3. **答案生成**
   - 结合检索结果和问题
   - 生成准确、相关的回答
   - 支持多轮对话

4. **答案溯源**
   - 显示参考文档来源
   - 提供相关度评分
   - 支持原文查看

## 配置说明

在 `.env` 文件中配置：

```env
# API密钥
OPENAI_API_KEY=your_api_key

# 向量库配置
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=5

# 数据库路径
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

## 性能优化建议

1. **文档预处理**
   - 清理无关内容
   - 统一文档格式
   - 合适的分块大小

2. **检索优化**
   - 调整检索数量
   - 优化查询重写
   - 使用混合检索

3. **生成优化**
   - 优化提示词模板
   - 调整上下文长度
   - 使用流式输出

## 扩展功能

- [ ] 支持更多文档格式
- [ ] 多语言支持
- [ ] 知识图谱集成
- [ ] 对话历史管理
- [ ] 批量问答处理
- [ ] API接口封装

## 许可证

MIT License

## 贡献指南

欢迎提交Issues和Pull Requests来改进项目。

## 联系信息

如有问题，请联系：[your-email@example.com]
