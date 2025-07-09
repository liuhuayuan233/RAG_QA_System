#!/usr/bin/env python3
"""
向量库构建脚本
用于处理文档并构建向量知识库
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.utils import setup_logging, create_directories

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="构建RAG向量知识库")
    parser.add_argument("--documents", "-d", type=str, default="./documents",
                       help="文档目录路径")
    parser.add_argument("--rebuild", "-r", action="store_true",
                       help="重建向量库（删除现有数据）")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger = setup_logging()
    
    try:
        # 验证配置
        config = Config()
        config.validate_config()
        
        logger.info("开始构建向量知识库")
        logger.info(f"文档目录: {args.documents}")
        logger.info(f"向量库目录: {config.CHROMA_PERSIST_DIRECTORY}")
        
        # 检查文档目录
        if not os.path.exists(args.documents):
            logger.error(f"文档目录不存在: {args.documents}")
            return 1
        
        # 创建必要的目录
        create_directories(config.CHROMA_PERSIST_DIRECTORY)
        
        # 初始化组件
        logger.info("初始化文档处理器...")
        doc_processor = DocumentProcessor()
        
        logger.info("初始化向量存储...")
        vector_store = VectorStore()
        
        # 如果需要重建，删除现有集合
        if args.rebuild:
            logger.info("删除现有向量库...")
            vector_store.delete_collection()
            # 重新初始化
            vector_store = VectorStore()
        
        # 处理文档
        logger.info("开始处理文档...")
        documents = doc_processor.process_directory(args.documents)
        
        if not documents:
            logger.warning("没有找到有效的文档")
            return 1
        
        # 验证文档质量
        logger.info("验证文档质量...")
        documents = doc_processor.validate_documents(documents)
        
        if not documents:
            logger.error("没有有效的文档通过验证")
            return 1
        
        # 添加到向量库
        logger.info("添加文档到向量库...")
        success = vector_store.add_documents(documents)
        
        if not success:
            logger.error("向量库构建失败")
            return 1
        
        # 显示统计信息
        info = vector_store.get_collection_info()
        logger.info("向量库构建完成!")
        logger.info(f"集合名称: {info.get('collection_name', 'N/A')}")
        logger.info(f"文档数量: {info.get('document_count', 'N/A')}")
        logger.info(f"嵌入模型: {info.get('embedding_model', 'N/A')}")
        
        # 测试检索
        logger.info("测试检索功能...")
        test_queries = ["这是什么文档？", "主要内容是什么？"]
        
        for query in test_queries:
            results = vector_store.similarity_search(query, k=3)
            logger.info(f"查询 '{query}' 返回 {len(results)} 个结果")
            
            for i, result in enumerate(results[:2]):
                logger.info(f"  结果{i+1}: {result['filename']} (得分: {result['score']:.3f})")
        
        logger.info("向量库构建和测试完成!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        return 1
    except Exception as e:
        logger.error(f"构建过程出错: {str(e)}")
        return 1

def check_documents_directory(directory: str) -> bool:
    """检查文档目录并显示信息"""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(directory):
        logger.error(f"文档目录不存在: {directory}")
        return False
    
    # 统计文件信息
    total_files = 0
    supported_files = 0
    file_types = {}
    
    config = Config()
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            total_files += 1
            ext = Path(file).suffix.lower()
            
            if ext in config.SUPPORTED_EXTENSIONS:
                supported_files += 1
            
            file_types[ext] = file_types.get(ext, 0) + 1
    
    logger.info(f"文档目录统计:")
    logger.info(f"  总文件数: {total_files}")
    logger.info(f"  支持的文件数: {supported_files}")
    logger.info(f"  文件类型分布:")
    
    for ext, count in sorted(file_types.items()):
        supported = "✓" if ext in config.SUPPORTED_EXTENSIONS else "✗"
        logger.info(f"    {ext}: {count} 个 {supported}")
    
    return supported_files > 0

def interactive_mode():
    """交互模式"""
    logger = setup_logging()
    
    print("=" * 50)
    print("RAG 向量知识库构建工具")
    print("=" * 50)
    
    # 获取文档目录
    default_docs_dir = "./documents"
    docs_dir = input(f"请输入文档目录路径 (默认: {default_docs_dir}): ").strip()
    if not docs_dir:
        docs_dir = default_docs_dir
    
    # 检查文档目录
    if not check_documents_directory(docs_dir):
        print("❌ 文档目录检查失败")
        return 1
    
    # 询问是否重建
    rebuild = input("是否重建向量库？(y/N): ").strip().lower() == 'y'
    
    # 构建参数
    sys.argv = ["build_vector_store.py", "--documents", docs_dir]
    if rebuild:
        sys.argv.append("--rebuild")
    
    print("\n开始构建向量库...")
    return main()

if __name__ == "__main__":
    # 如果没有命令行参数，启动交互模式
    if len(sys.argv) == 1:
        exit(interactive_mode())
    else:
        exit(main())
