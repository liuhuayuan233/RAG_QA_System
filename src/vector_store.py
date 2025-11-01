import os
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document as LangchainDocument
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd
from config.config import Config
from src.utils import setup_logging, safe_execute

# 设置环境变量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

logger = setup_logging()

class VectorStore:
    """向量存储管理器"""
    
    def __init__(self):
        self.config = Config()
        self.embeddings = None
        self.vector_store = None
        self.client = None
        self._initialize_embeddings()
        self._initialize_vector_store()
    
    def _initialize_embeddings(self):
        """初始化嵌入模型"""
        try:
            logger.info(f"正在加载嵌入模型: {self.config.EMBEDDING_MODEL}")
            
            # 简化的模型配置
            model_kwargs = {
                'device': 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1' else 'cpu'
            }
            
            encode_kwargs = {
                'normalize_embeddings': True,
                'batch_size': 32
            }
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.EMBEDDING_MODEL,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            
            logger.info("嵌入模型加载成功 (CUDA)")
            
        except Exception as e:
            logger.error(f"嵌入模型加载失败: {str(e)}")
            raise
    
    def _initialize_vector_store(self):
        """初始化向量数据库"""
        try:
            # 创建ChromaDB客户端
            self.client = chromadb.PersistentClient(
                path=self.config.CHROMA_PERSIST_DIRECTORY,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # 初始化Chroma向量存储
            self.vector_store = Chroma(
                collection_name=self.config.COLLECTION_NAME,
                embedding_function=self.embeddings,
                client=self.client,
                persist_directory=self.config.CHROMA_PERSIST_DIRECTORY
            )
            
            logger.info("向量数据库初始化成功")
            
        except Exception as e:
            logger.error(f"向量数据库初始化失败: {str(e)}")
            raise
    
    def add_documents(self, documents: List[LangchainDocument]) -> bool:
        """添加文档到向量库"""
        try:
            if not documents:
                logger.warning("没有文档需要添加")
                return False
            
            logger.info(f"正在添加 {len(documents)} 个文档到向量库")
            
            # 批量添加文档
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                self.vector_store.add_documents(batch)
                logger.info(f"已添加 {min(i + batch_size, len(documents))}/{len(documents)} 个文档")
            
            # 持久化存储
            self.vector_store.persist()
            logger.info("文档添加完成并已持久化")
            
            return True
            
        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """相似性搜索"""
        try:
            if k is None:
                k = self.config.TOP_K_RETRIEVAL
            
            # 执行相似性搜索
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            # 格式化结果
            formatted_results = []
            for doc, score in results:
                result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                    "filename": doc.metadata.get("filename", "未知文档"),
                    "source": doc.metadata.get("source", ""),
                    "chunk_id": doc.metadata.get("chunk_id", 0)
                }
                formatted_results.append(result)
            
            logger.info(f"检索到 {len(formatted_results)} 个相关文档")
            return formatted_results
            
        except Exception as e:
            logger.error(f"相似性搜索失败: {str(e)}")
            return []
    
    def similarity_search_with_threshold(self, query: str, k: int = None, threshold: float = None) -> List[Dict[str, Any]]:
        """基于阈值的相似性搜索"""
        if threshold is None:
            threshold = self.config.SIMILARITY_THRESHOLD
        
        results = self.similarity_search(query, k)
        
        # 过滤低相关度结果
        filtered_results = [r for r in results if r["score"] >= threshold]
        
        logger.info(f"阈值过滤后保留 {len(filtered_results)} 个高相关度文档")
        return filtered_results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        try:
            collection = self.client.get_collection(self.config.COLLECTION_NAME)
            count = collection.count()
            
            info = {
                "collection_name": self.config.COLLECTION_NAME,
                "document_count": count,
                "embedding_model": self.config.EMBEDDING_MODEL,
                "persist_directory": self.config.CHROMA_PERSIST_DIRECTORY
            }
            
            return info
            
        except Exception as e:
            logger.error(f"获取集合信息失败: {str(e)}")
            return {}
    
