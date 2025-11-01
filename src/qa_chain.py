import logging
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
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
                streaming=True  # 可以设置为True启用流式输出
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
            response = self.llm.invoke(prompt).content
            
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
    
    def clear_history(self):
        """清空对话历史"""
        self.chat_history = []
        logger.info("对话历史已清空")
    
