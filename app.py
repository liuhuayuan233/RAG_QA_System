import streamlit as st
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from config.config import Config
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.qa_chain import QAChain
from src.utils import setup_logging, format_sources

# 设置页面配置
st.set_page_config(
    page_title="RAG知识问答系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 隐藏Streamlit默认样式
st.markdown("""
<style>
.main > div {
    max-width: 1200px;
    margin: 0 auto;
}
.stAlert {
    margin-top: 1rem;
}
.source-card {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    background-color: #f8f9fa;
}
.score-badge {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.875rem;
    font-weight: 500;
}
.score-high {
    background-color: #d4edda;
    color: #155724;
}
.score-medium {
    background-color: #fff3cd;
    color: #856404;
}
.score-low {
    background-color: #f8d7da;
    color: #721c24;
}
</style>
""", unsafe_allow_html=True)

# 初始化session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.qa_chain = None
    st.session_state.chat_history = []
    st.session_state.vector_store_info = {}

def initialize_system():
    """初始化系统"""
    try:
        with st.spinner("正在初始化系统..."):
            # 验证配置
            config = Config()
            config.validate_config()
            
            # 初始化组件
            vector_store = VectorStore()
            retriever = Retriever(vector_store)
            qa_chain = QAChain(retriever)
            
            # 获取向量库信息
            vector_store_info = vector_store.get_collection_info()
            
            # 保存到session state
            st.session_state.qa_chain = qa_chain
            st.session_state.vector_store_info = vector_store_info
            st.session_state.initialized = True
            
            return True
            
    except Exception as e:
        st.error(f"系统初始化失败: {str(e)}")
        return False

def display_vector_store_info():
    """显示向量库信息"""
    if st.session_state.vector_store_info:
        info = st.session_state.vector_store_info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("文档数量", info.get('document_count', 0))
        
        with col2:
            st.metric("嵌入模型", info.get('embedding_model', 'N/A'))
        
        with col3:
            st.metric("集合名称", info.get('collection_name', 'N/A'))

def upload_and_process_document():
    """上传并处理文档"""
    st.subheader("📄 文档上传")
    
    uploaded_file = st.file_uploader(
        "选择文档文件",
        type=['pdf', 'docx', 'txt', 'md'],
        help="支持 PDF、Word、TXT、Markdown 格式"
    )
    
    if uploaded_file is not None:
        # 保存上传的文件
        documents_dir = Path("./documents")
        documents_dir.mkdir(exist_ok=True)
        
        file_path = documents_dir / uploaded_file.name
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"文件 '{uploaded_file.name}' 上传成功！")
        
        # 处理文档
        if st.button("处理文档"):
            try:
                with st.spinner("正在处理文档..."):
                    doc_processor = DocumentProcessor()
                    documents = doc_processor.process_document(str(file_path))
                    
                    if documents:
                        # 添加到向量库
                        vector_store = st.session_state.qa_chain.retriever.vector_store
                        success = vector_store.add_documents(documents)
                        
                        if success:
                            st.success(f"文档处理完成！生成了 {len(documents)} 个文档块。")
                            # 更新向量库信息
                            st.session_state.vector_store_info = vector_store.get_collection_info()
                            st.rerun()
                        else:
                            st.error("文档添加到向量库失败")
                    else:
                        st.error("文档处理失败")
                        
            except Exception as e:
                st.error(f"文档处理出错: {str(e)}")

def display_chat_interface():
    """显示聊天界面"""
    st.subheader("💬 智能问答")
    
    # 显示聊天历史
    if st.session_state.chat_history:
        st.markdown("### 对话历史")
        for i, chat in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**Q{i+1}:** {chat['question']}")
                st.markdown(f"**A{i+1}:** {chat['answer']}")
                
                # 显示源文档
                if chat.get('sources'):
                    with st.expander("📚 参考文档"):
                        display_sources(chat['sources'])
                
                st.divider()
    
    # 输入问题
    question = st.text_input(
        "请输入您的问题:",
        placeholder="例如：这些文档主要讲述了什么内容？",
        key="question_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        ask_button = st.button("🔍 提问", type="primary")
    
    with col2:
        clear_button = st.button("🗑️ 清空历史")
    
    # 处理提问
    if ask_button and question:
        try:
            with st.spinner("正在思考..."):
                result = st.session_state.qa_chain.ask(question)
                
                # 添加到历史记录
                st.session_state.chat_history.append(result)
                
                # 重新运行以更新界面
                st.rerun()
                
        except Exception as e:
            st.error(f"问答失败: {str(e)}")
    
    # 清空历史
    if clear_button:
        st.session_state.chat_history = []
        st.session_state.qa_chain.clear_history()
        st.rerun()

def display_sources(sources):
    """显示源文档信息"""
    if not sources:
        st.info("没有找到相关文档")
        return
    
    for i, source in enumerate(sources):
        score = source.get('score', 0.0)
        
        # 根据得分设置样式
        if score >= 0.8:
            score_class = "score-high"
        elif score >= 0.6:
            score_class = "score-medium"
        else:
            score_class = "score-low"
        
        st.markdown(f"""
        <div class="source-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <strong>📄 {source.get('filename', '未知文档')}</strong>
                <span class="score-badge {score_class}">相关度: {score:.3f}</span>
            </div>
            <div style="font-size: 0.9rem; color: #666;">
                {source.get('content_preview', source.get('content', ''))[:300]}...
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_advanced_search():
    """显示高级搜索"""
    st.subheader("🔍 高级搜索")
    
    # 搜索选项
    search_type = st.selectbox(
        "搜索类型",
        ["语义搜索", "关键词搜索", "文档搜索"]
    )
    
    if search_type == "语义搜索":
        query = st.text_input("输入查询内容:")
        k = st.slider("返回结果数量", 1, 20, 5)
        
        if st.button("搜索") and query:
            try:
                with st.spinner("搜索中..."):
                    retriever = st.session_state.qa_chain.retriever
                    results = retriever.retrieve(query, k)
                    
                    if results:
                        st.success(f"找到 {len(results)} 个相关结果")
                        display_sources(results)
                    else:
                        st.warning("没有找到相关结果")
                        
            except Exception as e:
                st.error(f"搜索失败: {str(e)}")
    
    elif search_type == "关键词搜索":
        keywords = st.text_input("输入关键词（用空格分隔）:")
        
        if st.button("搜索") and keywords:
            try:
                with st.spinner("搜索中..."):
                    retriever = st.session_state.qa_chain.retriever
                    keyword_list = keywords.split()
                    results = retriever.retrieve_by_keywords(keyword_list)
                    
                    if results:
                        st.success(f"找到 {len(results)} 个相关结果")
                        display_sources(results)
                    else:
                        st.warning("没有找到相关结果")
                        
            except Exception as e:
                st.error(f"搜索失败: {str(e)}")
    
    elif search_type == "文档搜索":
        # 获取所有文档列表
        if st.session_state.vector_store_info:
            filename = st.text_input("输入文档名称:")
            
            if st.button("搜索") and filename:
                try:
                    with st.spinner("搜索中..."):
                        retriever = st.session_state.qa_chain.retriever
                        results = retriever.retrieve_by_document(filename)
                        
                        if results:
                            st.success(f"找到 {len(results)} 个文档块")
                            display_sources(results)
                        else:
                            st.warning("没有找到相关文档")
                            
                except Exception as e:
                    st.error(f"搜索失败: {str(e)}")

def display_system_stats():
    """显示系统统计"""
    st.subheader("📊 系统统计")
    
    # 对话统计
    total_chats = len(st.session_state.chat_history)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("总对话数", total_chats)
    
    with col2:
        if st.session_state.vector_store_info:
            st.metric("知识库文档数", st.session_state.vector_store_info.get('document_count', 0))
    
    # 最近对话
    if st.session_state.chat_history:
        st.markdown("### 最近对话")
        recent_chats = st.session_state.chat_history[-3:]
        
        for chat in recent_chats:
            with st.expander(f"Q: {chat['question'][:50]}..."):
                st.write(f"**问题:** {chat['question']}")
                st.write(f"**回答:** {chat['answer']}")
                if chat.get('sources'):
                    st.write(f"**参考文档数:** {len(chat['sources'])}")

def main():
    """主函数"""
    st.title("🤖 RAG 知识问答系统")
    st.markdown("基于检索增强生成的智能问答系统")
    
    # 初始化系统
    if not st.session_state.initialized:
        if not initialize_system():
            st.stop()
    
    # 侧边栏
    with st.sidebar:
        st.header("系统信息")
        display_vector_store_info()
        
        st.header("功能选择")
        page = st.selectbox(
            "选择功能",
            ["💬 智能问答", "📄 文档上传", "🔍 高级搜索", "📊 系统统计"]
        )
        
        # 配置选项
        st.header("配置")
        if st.button("🔄 重新初始化"):
            st.session_state.initialized = False
            st.rerun()
        
        # 导出对话历史
        if st.session_state.chat_history:
            if st.button("💾 导出对话历史"):
                export_data = {
                    "export_time": datetime.now().isoformat(),
                    "chat_history": st.session_state.chat_history
                }
                
                st.download_button(
                    label="下载 JSON 文件",
                    data=json.dumps(export_data, ensure_ascii=False, indent=2),
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    # 主内容区域
    if page == "💬 智能问答":
        display_chat_interface()
    elif page == "📄 文档上传":
        upload_and_process_document()
    elif page == "🔍 高级搜索":
        display_advanced_search()
    elif page == "📊 系统统计":
        display_system_stats()
    
    # 底部信息
    st.markdown("---")
    st.markdown(
        "💡 **提示:** 系统基于BGE中文嵌入模型和DeepSeek-R1进行问答，支持多种文档格式的上传和处理。"
    )

if __name__ == "__main__":
    main()
