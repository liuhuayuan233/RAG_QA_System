#!/usr/bin/env python3
"""
RAG系统测试脚本
用于测试各个组件的功能
"""

import sys
import os
from pathlib import Path
import logging

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.qa_chain import QAChain
from src.utils import setup_logging

def test_document_processor():
    """测试文档处理器"""
    print("=" * 50)
    print("测试文档处理器")
    print("=" * 50)
    
    try:
        processor = DocumentProcessor()
        
        # 测试处理示例文档
        docs_dir = "./documents"
        if os.path.exists(docs_dir):
            documents = processor.process_directory(docs_dir)
            
            print(f"✅ 成功处理 {len(documents)} 个文档块")
            
            # 显示前几个文档的信息
            for i, doc in enumerate(documents[:3]):
                print(f"\n文档块 {i+1}:")
                print(f"  来源: {doc.metadata.get('filename', 'N/A')}")
                print(f"  内容长度: {len(doc.page_content)}")
                print(f"  内容预览: {doc.page_content[:100]}...")
                
            return documents
        else:
            print("❌ 文档目录不存在")
            return []
            
    except Exception as e:
        print(f"❌ 文档处理失败: {str(e)}")
        return []

def test_vector_store(documents):
    """测试向量存储"""
    print("\n" + "=" * 50)
    print("测试向量存储")
    print("=" * 50)
    
    try:
        vector_store = VectorStore()
        
        # 添加文档
        if documents:
            success = vector_store.add_documents(documents)
            
            if success:
                print("✅ 文档添加到向量库成功")
                
                # 获取向量库信息
                info = vector_store.get_collection_info()
                print(f"  文档数量: {info.get('document_count', 'N/A')}")
                print(f"  嵌入模型: {info.get('embedding_model', 'N/A')}")
                
                # 测试检索
                test_queries = [
                    "什么是人工智能？",
                    "机器学习算法有哪些？",
                    "深度学习框架比较"
                ]
                
                for query in test_queries:
                    results = vector_store.similarity_search(query, k=3)
                    print(f"\n查询: {query}")
                    print(f"  返回结果数: {len(results)}")
                    
                    for i, result in enumerate(results):
                        print(f"  结果{i+1}: {result['filename']} (得分: {result['score']:.3f})")
                
                return vector_store
            else:
                print("❌ 文档添加失败")
                return None
        else:
            print("❌ 没有文档可添加")
            return None
            
    except Exception as e:
        print(f"❌ 向量存储测试失败: {str(e)}")
        return None

def test_retriever(vector_store):
    """测试检索器"""
    print("\n" + "=" * 50)
    print("测试检索器")
    print("=" * 50)
    
    try:
        retriever = Retriever(vector_store)
        
        # 测试不同的检索方法
        test_queries = [
            "人工智能的分类",
            "深度学习和机器学习的区别",
            "TensorFlow和PyTorch的比较"
        ]
        
        for query in test_queries:
            print(f"\n查询: {query}")
            
            # 基本检索
            results = retriever.retrieve(query, k=3)
            print(f"  基本检索结果数: {len(results)}")
            
            # 获取用于生成的上下文
            context = retriever.get_context_for_generation(query)
            print(f"  上下文长度: {len(context)}")
            
            # 获取源信息
            source_info = retriever.get_source_info(query)
            print(f"  源文档数: {len(source_info)}")
            
            # 显示最相关的结果
            if results:
                best_result = results[0]
                print(f"  最相关文档: {best_result['filename']}")
                print(f"  相关度得分: {best_result.get('final_score', best_result['score']):.3f}")
        
        return retriever
        
    except Exception as e:
        print(f"❌ 检索器测试失败: {str(e)}")
        return None

def test_qa_chain(retriever):
    """测试问答链"""
    print("\n" + "=" * 50)
    print("测试问答链")
    print("=" * 50)
    
    try:
        qa_chain = QAChain(retriever)
        
        # 测试问答
        test_questions = [
            "人工智能有哪些主要分类？",
            "深度学习中常用的激活函数有哪些？",
            "TensorFlow框架有什么优势？"
        ]
        
        for question in test_questions:
            print(f"\n问题: {question}")
            
            try:
                result = qa_chain.ask(question)
                
                print(f"答案: {result['answer']}")
                print(f"参考文档数: {len(result['sources'])}")
                
                # 显示源文档
                if result['sources']:
                    print("参考文档:")
                    for i, source in enumerate(result['sources'][:2]):
                        print(f"  {i+1}. {source['filename']} (相关度: {source['score']:.3f})")
                
            except Exception as e:
                print(f"❌ 问答失败: {str(e)}")
                continue
        
        return qa_chain
        
    except Exception as e:
        print(f"❌ 问答链测试失败: {str(e)}")
        return None

def test_end_to_end():
    """端到端测试"""
    print("\n" + "=" * 50)
    print("端到端测试")
    print("=" * 50)
    
    try:
        # 验证配置
        config = Config()
        config.validate_config()
        print("✅ 配置验证成功")
        
        # 测试文档处理
        documents = test_document_processor()
        if not documents:
            print("❌ 文档处理失败，终止测试")
            return False
        
        # 测试向量存储
        vector_store = test_vector_store(documents)
        if not vector_store:
            print("❌ 向量存储失败，终止测试")
            return False
        
        # 测试检索器
        retriever = test_retriever(vector_store)
        if not retriever:
            print("❌ 检索器失败，终止测试")
            return False
        
        # 测试问答链
        qa_chain = test_qa_chain(retriever)
        if not qa_chain:
            print("❌ 问答链失败，终止测试")
            return False
        
        print("\n" + "=" * 50)
        print("✅ 所有测试通过！RAG系统工作正常")
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"\n❌ 端到端测试失败: {str(e)}")
        return False

def interactive_test():
    """交互式测试"""
    print("\n" + "=" * 50)
    print("交互式测试模式")
    print("=" * 50)
    
    try:
        # 初始化系统
        config = Config()
        config.validate_config()
        
        vector_store = VectorStore()
        retriever = Retriever(vector_store)
        qa_chain = QAChain(retriever)
        
        print("系统初始化完成！")
        print("输入 'quit' 退出，输入 'help' 查看帮助")
        
        while True:
            question = input("\n请输入您的问题: ").strip()
            
            if question.lower() == 'quit':
                print("再见！")
                break
            elif question.lower() == 'help':
                print("帮助信息:")
                print("- 输入问题进行问答")
                print("- 输入 'quit' 退出")
                print("- 输入 'history' 查看对话历史")
                print("- 输入 'clear' 清空对话历史")
                continue
            elif question.lower() == 'history':
                history = qa_chain.get_history()
                if history:
                    print("对话历史:")
                    for i, chat in enumerate(history):
                        print(f"{i+1}. Q: {chat['question']}")
                        print(f"   A: {chat['answer'][:100]}...")
                else:
                    print("暂无对话历史")
                continue
            elif question.lower() == 'clear':
                qa_chain.clear_history()
                print("对话历史已清空")
                continue
            elif not question:
                continue
            
            try:
                print("🤔 思考中...")
                result = qa_chain.ask(question)
                
                print(f"\n🤖 回答: {result['answer']}")
                
                if result['sources']:
                    print(f"\n📚 参考文档 ({len(result['sources'])} 个):")
                    for i, source in enumerate(result['sources'][:3]):
                        print(f"  {i+1}. {source['filename']} (相关度: {source['score']:.3f})")
                
            except Exception as e:
                print(f"❌ 回答失败: {str(e)}")
                
    except Exception as e:
        print(f"❌ 交互式测试失败: {str(e)}")

def main():
    """主函数"""
    # 设置日志
    logger = setup_logging()
    
    print("RAG 知识问答系统测试工具")
    print("=" * 60)
    
    # 检查参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            interactive_test()
        elif sys.argv[1] == "e2e":
            test_end_to_end()
        else:
            print("用法: python test_system.py [interactive|e2e]")
    else:
        # 默认运行端到端测试
        success = test_end_to_end()
        
        if success:
            # 询问是否进入交互模式
            choice = input("\n是否进入交互式测试模式？(y/N): ").strip().lower()
            if choice == 'y':
                interactive_test()

if __name__ == "__main__":
    main()
