import logging
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from src.vector_store import VectorStore
from src.utils import setup_logging, calculate_relevance_score

logger = setup_logging()

class Retriever:
    """检索器 - 负责从向量库中检索相关文档"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.config = vector_store.config
    
    def retrieve(self, query: str, k: int = None, use_threshold: bool = True) -> List[Dict[str, Any]]:
        """检索相关文档"""
        try:
            if k is None:
                k = self.config.TOP_K_RETRIEVAL
            
            logger.info(f"正在检索查询: {query}")
            
            # 使用阈值过滤
            if use_threshold:
                results = self.vector_store.similarity_search_with_threshold(
                    query=query,
                    k=k,
                    threshold=self.config.SIMILARITY_THRESHOLD
                )
            else:
                results = self.vector_store.similarity_search(query=query, k=k)
            
            # 重新排序结果
            results = self._rerank_results(query, results)
            
            logger.info(f"检索完成，返回 {len(results)} 个相关文档")
            return results
            
        except Exception as e:
            logger.error(f"检索失败: {str(e)}")
            return []
    
    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """重新排序检索结果"""
        try:
            # 计算额外的相关性得分
            for result in results:
                content = result["content"]
                
                # 计算词汇重叠度
                lexical_score = calculate_relevance_score(query, content)
                
                # 计算长度惩罚（太短或太长的文档得分降低）
                length_penalty = self._calculate_length_penalty(content)
                
                # 计算位置奖励（文档开头的内容得分提高）
                position_bonus = self._calculate_position_bonus(result.get("metadata", {}))
                
                # 综合得分
                final_score = (
                    result["score"] * 0.7 +  # 语义相似度
                    lexical_score * 0.2 +    # 词汇相似度
                    length_penalty * 0.05 +  # 长度惩罚
                    position_bonus * 0.05    # 位置奖励
                )
                
                result["final_score"] = final_score
                result["lexical_score"] = lexical_score
                result["length_penalty"] = length_penalty
                result["position_bonus"] = position_bonus
            
            # 按最终得分排序
            results.sort(key=lambda x: x["final_score"], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"重新排序失败: {str(e)}")
            return results
    
    def _calculate_length_penalty(self, content: str) -> float:
        """计算长度惩罚"""
        length = len(content)
        
        # 理想长度范围
        ideal_min = 200
        ideal_max = 1000
        
        if ideal_min <= length <= ideal_max:
            return 1.0
        elif length < ideal_min:
            return length / ideal_min
        else:
            return ideal_max / length
    
    def _calculate_position_bonus(self, metadata: Dict[str, Any]) -> float:
        """计算位置奖励"""
        chunk_id = metadata.get("chunk_id", 0)
        
        # 文档开头的块得分更高
        if chunk_id == 0:
            return 1.0
        elif chunk_id <= 2:
            return 0.8
        elif chunk_id <= 5:
            return 0.6
        else:
            return 0.4
    
    def get_context_for_generation(self, query: str, max_context_length: int = 4000) -> str:
        """获取用于生成的上下文"""
        try:
            # 检索相关文档
            results = self.retrieve(query)
            
            if not results:
                return "没有找到相关文档。"
            
            # 构建上下文
            context_parts = []
            current_length = 0
            
            for i, result in enumerate(results):
                content = result["content"]
                filename = result["filename"]
                score = result.get("final_score", result["score"])
                
                # 格式化文档片段
                doc_part = f"[文档{i+1}: {filename} (相关度: {score:.3f})]\n{content}\n"
                
                # 检查长度限制
                if current_length + len(doc_part) > max_context_length:
                    if current_length == 0:  # 第一个文档就超长
                        # 截断内容
                        available_length = max_context_length - len(f"[文档1: {filename} (相关度: {score:.3f})]\n\n")
                        truncated_content = content[:available_length] + "..."
                        doc_part = f"[文档1: {filename} (相关度: {score:.3f})]\n{truncated_content}\n"
                        context_parts.append(doc_part)
                    break
                
                context_parts.append(doc_part)
                current_length += len(doc_part)
            
            context = "\n".join(context_parts)
            
            logger.info(f"构建上下文完成，长度: {len(context)}")
            return context
            
        except Exception as e:
            logger.error(f"获取上下文失败: {str(e)}")
            return "获取上下文时出现错误。"
    
    def get_source_info(self, query: str) -> List[Dict[str, Any]]:
        """获取源文档信息"""
        try:
            results = self.retrieve(query)
            
            source_info = []
            for result in results:
                info = {
                    "filename": result["filename"],
                    "source": result.get("source", ""),
                    "chunk_id": result.get("chunk_id", 0),
                    "score": result.get("final_score", result["score"]),
                    "content_preview": result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"]
                }
                source_info.append(info)
            
            return source_info
            
        except Exception as e:
            logger.error(f"获取源信息失败: {str(e)}")
            return []
