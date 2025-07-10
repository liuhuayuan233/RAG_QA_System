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
    LLM_MODEL = "deepseek-ai/DeepSeek-R1"  
    
    # 向量数据库配置
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "1024"))
    COLLECTION_NAME = "documents"
    
    # 文档处理配置 - 安全的配置解析
    @staticmethod
    def _safe_int(value, default):
        """安全地解析整数配置，去除注释"""
        try:
            # 去除注释部分
            if isinstance(value, str) and '#' in value:
                value = value.split('#')[0].strip()
            return int(value)
        except (ValueError, TypeError):
            return default
    
    MAX_DOCUMENT_SIZE = _safe_int.__func__(os.getenv("MAX_DOCUMENT_SIZE", "10000000"), 10000000)
    CHUNK_SIZE = _safe_int.__func__(os.getenv("CHUNK_SIZE", "1000"), 1000)
    CHUNK_OVERLAP = _safe_int.__func__(os.getenv("CHUNK_OVERLAP", "200"), 200)
    
    # 检索配置
    TOP_K_RETRIEVAL = _safe_int.__func__(os.getenv("TOP_K_RETRIEVAL", "5"), 5)
    SIMILARITY_THRESHOLD = 0.7
    
    # 支持的文档格式
    SUPPORTED_EXTENSIONS = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".txt": "txt",
        ".md": "markdown",
        ".jsonl": "jsonl"
    }
    
    # 文档目录
    DOCUMENTS_DIR = "./documents"
    
    # 系统提示词
    SYSTEM_PROMPT = """你是一个专业的医疗知识问答助手。请基于给定的医疗文档内容回答用户的问题。

要求：
1. 回答要准确、专业、有针对性
2. 优先使用检索到的医疗文档内容
3. 如果文档内容不足以回答问题，请明确说明
4. 提供的医疗建议仅供参考，不能替代专业医生的诊断
5. 对于严重症状，建议用户及时就医
6. 回答结束后请标注参考的医疗文档来源

重要提醒：本系统提供的信息仅供健康教育和参考，不能替代专业医疗诊断和治疗。如有严重症状或疑虑，请及时就医。

医疗文档内容：
{context}

用户问题：{question}

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
