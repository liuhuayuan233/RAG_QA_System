#!/usr/bin/env python3
"""
最简化的CUDA向量库构建脚本
"""

import os
import sys
from pathlib import Path

# 设置环境变量，避免不必要的输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

def main():
    print("🚀 CUDA向量库构建")
    print("=" * 40)
    
    try:
        # 直接导入必要的模块
        from config.config import Config
        from src.document_processor import DocumentProcessor
        from src.vector_store import VectorStore
        
        config = Config()
        
        # 检查文档
        docs_dir = config.DOCUMENTS_DIR
        if not os.path.exists(docs_dir) or not os.listdir(docs_dir):
            print("❌ 请先添加文档到 documents/ 目录")
            print("💡 或运行: python scripts/download_datasets.py --medical --limit 1000")
            return
        
        # 处理文档
        print("📖 处理文档...")
        processor = DocumentProcessor()
        documents = processor.process_directory(docs_dir)
        
        if not documents:
            print("❌ 未找到有效文档")
            return
        
        print(f"✅ 处理了 {len(documents)} 个文档块")
        
        # 构建向量库
        print("🔍 构建CUDA向量库...")
        vector_store = VectorStore()
        vector_store.add_documents(documents)
        
        print("✅ 完成！向量库已保存")
        print("🚀 现在可以运行: streamlit run app.py")
        
    except ImportError as e:
        if "cv2" in str(e):
            print("❌ OpenCV冲突问题")
            print("🔧 解决方案:")
            print("pip install --upgrade --force-reinstall sentence-transformers")
            print("或者重新创建虚拟环境")
        else:
            print(f"❌ 导入错误: {e}")
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    main()
