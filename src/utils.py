import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import re

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('rag_system.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """清理文本内容"""
    if not text:
        return ""
    
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    # 移除特殊字符
    text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？；：""''（）【】《》]', '', text)
    # 移除过短的行
    lines = text.split('\n')
    lines = [line.strip() for line in lines if len(line.strip()) > 10]
    
    return '\n'.join(lines)

def validate_file_size(file_path: str, max_size: int = 10 * 1024 * 1024) -> bool:
    """验证文件大小"""
    try:
        size = os.path.getsize(file_path)
        return size <= max_size
    except OSError:
        return False

def get_file_extension(file_path: str) -> str:
    """获取文件扩展名"""
    return Path(file_path).suffix.lower()

def create_directories(*dirs: str) -> None:
    """创建目录"""
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def format_sources(sources: List[Dict[str, Any]]) -> str:
    """格式化文档来源信息"""
    if not sources:
        return "无参考文档"
    
    formatted_sources = []
    for i, source in enumerate(sources, 1):
        filename = source.get('filename', '未知文档')
        page = source.get('page', '')
        score = source.get('score', 0.0)
        
        source_info = f"{i}. 📄 {filename}"
        if page:
            source_info += f" (第{page}页)"
        source_info += f" - 相关度: {score:.2f}"
        
        formatted_sources.append(source_info)
    
    return "\n".join(formatted_sources)

def truncate_text(text: str, max_length: int = 500) -> str:
    """截断文本到指定长度"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def calculate_relevance_score(query: str, text: str) -> float:
    """计算文本相关性得分（简单实现）"""
    if not query or not text:
        return 0.0
    
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    
    intersection = query_words & text_words
    union = query_words | text_words
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

class TokenCounter:
    """Token计数器"""
    
    @staticmethod
    def count_tokens(text: str) -> int:
        """简单的token计数（中文按字符计算）"""
        # 简化的token计算，实际应用中建议使用tiktoken
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        return chinese_chars + english_words

    @staticmethod
    def truncate_by_tokens(text: str, max_tokens: int = 4000) -> str:
        """按token数量截断文本"""
        if TokenCounter.count_tokens(text) <= max_tokens:
            return text
        
        # 简单截断策略
        chars_per_token = len(text) / TokenCounter.count_tokens(text)
        target_length = int(max_tokens * chars_per_token)
        
        return text[:target_length]

def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """提取关键词（简单实现）"""
    if not text:
        return []
    
    # 简单的关键词提取
    words = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}', text.lower())
    
    # 统计词频
    word_count = {}
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    
    # 排序并返回前k个
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in sorted_words[:top_k]]

def safe_execute(func, *args, **kwargs):
    """安全执行函数"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logging.error(f"执行函数 {func.__name__} 时发生错误: {str(e)}")
        return None
