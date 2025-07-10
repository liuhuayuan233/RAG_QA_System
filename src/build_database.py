#!/usr/bin/env python3
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
        
        # 先验证配置
        print("🔧 验证配置...")
        try:
            config = Config()
            print(f"✅ 文档目录: {config.DOCUMENTS_DIR}")
            print(f"✅ 向量数据库目录: {config.CHROMA_PERSIST_DIRECTORY}")
            print(f"✅ 嵌入模型: {config.EMBEDDING_MODEL}")
        except Exception as config_error:
            print(f"❌ 配置错误: {config_error}")
            print("🔧 检查环境变量设置，可能包含无效的注释")
            return
        
        # 检查文档
        docs_dir = config.DOCUMENTS_DIR
        if not os.path.exists(docs_dir) or not os.listdir(docs_dir):
            print("❌ 请先添加文档到 documents/ 目录")
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
