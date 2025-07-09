import logging
from typing import List, Dict, Any, Optional
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.callbacks import StreamingStdOutCallbackHandler

from config.config import Config
from src.retriever import Retriever
from src.utils import setup_logging, format_sources, TokenCounter

logger = setup_logging()

class QAChain:
    """问答链 - 负责结合检索结果生成答案"""
    
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.config = Config()
        self.llm = None
        self.chat_history = []
        self._initialize_llm()
    
    def _initialize_llm(self):
        """初始化大语言模型"""
        try:
            logger.info(f"正在初始化LLM: {self.config.LLM_MODEL}")
            
            # 使用ChatOpenAI
            self.llm = ChatOpenAI(
                model_name=self.config.LLM_MODEL,
                temperature=0.1,  # 低温度以获得更准确的回答
                max_tokens=2000,
                openai_api_key=self.config.OPENAI_API_KEY,
                openai_api_base=self.config.OPENAI_API_BASE,
                streaming=False  # 可以设置为True启用流式输出
            )
            
            logger.info("LLM初始化成功")
            
        except Exception as e:
            logger.error(f"LLM初始化失败: {str(e)}")
            raise
    
    def ask(self, question: str, use_history: bool = True) -> Dict[str, Any]:
        """询问问题并获取答案"""
        try:
            logger.info(f"收到问题: {question}")
            
            # 检索相关文档
            context = self.retriever.get_context_for_generation(question)
            source_info = self.retriever.get_source_info(question)
            
            # 构建提示词
            prompt = self._build_prompt(question, context, use_history)
            
            # 生成答案
            response = self.llm.predict(prompt)
            
            # 记录对话历史
            if use_history:
                self.chat_history.append({
                    "question": question,
                    "answer": response,
                    "context": context,
                    "sources": source_info
                })
            
            # 构建返回结果
            result = {
                "question": question,
                "answer": response,
                "sources": source_info,
                "context": context,
                "source_summary": format_sources(source_info)
            }
            
            logger.info("问答完成")
            return result
            
        except Exception as e:
            logger.error(f"问答失败: {str(e)}")
            return {
                "question": question,
                "answer": f"抱歉，处理您的问题时出现错误: {str(e)}",
                "sources": [],
                "context": "",
                "source_summary": ""
            }
    
    def _build_prompt(self, question: str, context: str, use_history: bool = True) -> str:
        """构建提示词"""
        try:
            # 基础提示词
            prompt = self.config.SYSTEM_PROMPT.format(
                context=context,
                question=question
            )
            
            # 添加对话历史
            if use_history and self.chat_history:
                history_text = "\n\n=== 对话历史 ===\n"
                # 只保留最近的3轮对话
                recent_history = self.chat_history[-3:]
                
                for i, chat in enumerate(recent_history):
                    history_text += f"Q{i+1}: {chat['question']}\n"
                    history_text += f"A{i+1}: {chat['answer']}\n\n"
                
                prompt = history_text + prompt
            
            # 截断过长的提示词
            prompt = TokenCounter.truncate_by_tokens(prompt, 6000)
            
            return prompt
            
        except Exception as e:
            logger.error(f"构建提示词失败: {str(e)}")
            return self.config.SYSTEM_PROMPT.format(
                context=context[:2000],
                question=question
            )
    
    def ask_with_streaming(self, question: str, callback=None) -> Dict[str, Any]:
        """流式问答"""
        try:
            logger.info(f"流式问答: {question}")
            
            # 检索相关文档
            context = self.retriever.get_context_for_generation(question)
            source_info = self.retriever.get_source_info(question)
            
            # 构建提示词
            prompt = self._build_prompt(question, context)
            
            # 创建流式LLM
            streaming_llm = ChatOpenAI(
                model_name=self.config.LLM_MODEL,
                temperature=0.1,
                max_tokens=2000,
                openai_api_key=self.config.OPENAI_API_KEY,
                openai_api_base=self.config.OPENAI_API_BASE,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()] if callback is None else [callback]
            )
            
            # 生成答案
            response = streaming_llm.predict(prompt)
            
            # 记录对话历史
            self.chat_history.append({
                "question": question,
                "answer": response,
                "context": context,
                "sources": source_info
            })
            
            result = {
                "question": question,
                "answer": response,
                "sources": source_info,
                "context": context,
                "source_summary": format_sources(source_info)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"流式问答失败: {str(e)}")
            return {
                "question": question,
                "answer": f"抱歉，处理您的问题时出现错误: {str(e)}",
                "sources": [],
                "context": "",
                "source_summary": ""
            }
    
    def multi_turn_ask(self, questions: List[str]) -> List[Dict[str, Any]]:
        """多轮问答"""
        try:
            results = []
            
            for i, question in enumerate(questions):
                logger.info(f"处理第 {i+1} 个问题: {question}")
                
                result = self.ask(question, use_history=True)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"多轮问答失败: {str(e)}")
            return []
    
    def clear_history(self):
        """清空对话历史"""
        self.chat_history = []
        logger.info("对话历史已清空")
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取对话历史"""
        return self.chat_history.copy()
    
    def explain_answer(self, question: str, answer: str) -> str:
        """解释答案的推理过程"""
        try:
            # 获取用于生成答案的上下文
            context = self.retriever.get_context_for_generation(question)
            
            explain_prompt = f"""
请解释以下问答的推理过程：

问题: {question}
答案: {answer}

参考文档: {context[:1000]}

请分析：
1. 答案是如何从文档中推导出来的
2. 使用了哪些关键信息
3. 推理的逻辑链条
4. 答案的可信度

解释：
"""
            
            explanation = self.llm.predict(explain_prompt)
            return explanation
            
        except Exception as e:
            logger.error(f"解释答案失败: {str(e)}")
            return "无法生成解释"
    
    def evaluate_answer_quality(self, question: str, answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估答案质量"""
        try:
            # 简单的质量评估
            quality_metrics = {
                "relevance": self._calculate_relevance(question, answer),
                "completeness": self._calculate_completeness(answer),
                "source_support": self._calculate_source_support(answer, sources),
                "clarity": self._calculate_clarity(answer),
                "confidence": self._calculate_confidence(sources)
            }
            
            # 计算总体得分
            total_score = sum(quality_metrics.values()) / len(quality_metrics)
            quality_metrics["total_score"] = total_score
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"评估答案质量失败: {str(e)}")
            return {"total_score": 0.5}
    
    def _calculate_relevance(self, question: str, answer: str) -> float:
        """计算回答相关性"""
        # 简单的关键词匹配
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        intersection = question_words & answer_words
        union = question_words | answer_words
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _calculate_completeness(self, answer: str) -> float:
        """计算回答完整性"""
        # 基于答案长度的简单评估
        length = len(answer)
        if length < 50:
            return 0.3
        elif length < 200:
            return 0.7
        else:
            return 1.0
    
    def _calculate_source_support(self, answer: str, sources: List[Dict[str, Any]]) -> float:
        """计算源文档支持度"""
        if not sources:
            return 0.0
        
        # 检查答案中是否包含源文档的内容
        support_score = 0.0
        for source in sources:
            if any(word in answer.lower() for word in source.get("content", "").lower().split()[:10]):
                support_score += source.get("score", 0.0)
        
        return min(support_score / len(sources), 1.0)
    
    def _calculate_clarity(self, answer: str) -> float:
        """计算回答清晰度"""
        # 简单的清晰度评估
        sentences = answer.split('。')
        if len(sentences) < 2:
            return 0.5
        
        avg_sentence_length = sum(len(s) for s in sentences) / len(sentences)
        
        # 理想句子长度20-50字符
        if 20 <= avg_sentence_length <= 50:
            return 1.0
        elif avg_sentence_length < 20:
            return 0.7
        else:
            return 0.8
    
    def _calculate_confidence(self, sources: List[Dict[str, Any]]) -> float:
        """计算置信度"""
        if not sources:
            return 0.0
        
        # 基于源文档的相关性得分
        scores = [source.get("score", 0.0) for source in sources]
        avg_score = sum(scores) / len(scores)
        
        return min(avg_score, 1.0)
