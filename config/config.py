import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # 如果没有安装python-dotenv，则跳过
    pass

class Config:
    """系统配置类"""
    
    # API配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    
    # 模型配置
    EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"  # BGE中文嵌入模型
    LLM_MODEL = "gpt-3.5-turbo"  # 可改为gpt-4或其他模型
    
    # 向量数据库配置
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "1024"))
    COLLECTION_NAME = "documents"
    
    # 文档处理配置
    MAX_DOCUMENT_SIZE = int(os.getenv("MAX_DOCUMENT_SIZE", "10000000"))  # 10MB
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # 检索配置
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))
    SIMILARITY_THRESHOLD = 0.7
    
    # 支持的文档格式
    SUPPORTED_EXTENSIONS = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".txt": "txt",
        ".md": "markdown"
    }
    
    # 文档目录
    DOCUMENTS_DIR = "./documents"
    
    # 系统提示词
    SYSTEM_PROMPT = """你是一个专业的知识问答助手。请基于给定的文档内容回答用户的问题。

要求：
1. 回答要准确、简洁、有针对性
2. 优先使用检索到的文档内容
3. 如果文档内容不足以回答问题，请明确说明
4. 回答结束后请标注参考的文档来源

文档内容：
{context}

问题：{question}

回答："""
    
    # Streamlit配置
    PAGE_TITLE = "RAG知识问答系统"
    PAGE_ICON = "🤖"
    
    @classmethod
    def validate_config(cls):
        """验证配置"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("请设置OPENAI_API_KEY环境变量")
        
        # 创建必要的目录
        os.makedirs(cls.DOCUMENTS_DIR, exist_ok=True)
        os.makedirs(cls.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        
        return True
