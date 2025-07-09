#!/usr/bin/env python3
"""
数据集质量评估工具
用于评估和分析数据集的质量和特征
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import jieba
import re

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from src.document_processor import DocumentProcessor
from src.utils import setup_logger, clean_text

logger = setup_logger()

class DatasetAnalyzer:
    """数据集分析器"""
    
    def __init__(self):
        self.processor = DocumentProcessor()
        self.stats = {}
    
    def analyze_directory(self, directory: str) -> Dict[str, Any]:
        """分析目录中的所有文档"""
        try:
            logger.info(f"开始分析目录: {directory}")
            
            # 获取所有文件
            files = []
            for root, dirs, filenames in os.walk(directory):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    files.append(file_path)
            
            # 分析每个文件
            file_stats = []
            total_size = 0
            total_words = 0
            
            for file_path in files:
                try:
                    stats = self.analyze_file(file_path)
                    if stats:
                        file_stats.append(stats)
                        total_size += stats['size_bytes']
                        total_words += stats['word_count']
                except Exception as e:
                    logger.warning(f"分析文件失败 {file_path}: {str(e)}")
                    continue
            
            # 计算总体统计
            overall_stats = {
                'total_files': len(file_stats),
                'total_size_mb': total_size / (1024 * 1024),
                'total_words': total_words,
                'average_file_size_kb': (total_size / len(file_stats)) / 1024 if file_stats else 0,
                'average_words_per_file': total_words / len(file_stats) if file_stats else 0
            }
            
            # 文件类型分布
            file_types = Counter([stats['file_type'] for stats in file_stats])
            
            # 内容质量分析
            quality_scores = [stats['quality_score'] for stats in file_stats]
            
            result = {
                'overall_stats': overall_stats,
                'file_stats': file_stats,
                'file_types': dict(file_types),
                'quality_distribution': {
                    'mean': np.mean(quality_scores) if quality_scores else 0,
                    'std': np.std(quality_scores) if quality_scores else 0,
                    'min': np.min(quality_scores) if quality_scores else 0,
                    'max': np.max(quality_scores) if quality_scores else 0
                }
            }
            
            self.stats = result
            logger.info(f"分析完成，共处理 {len(file_stats)} 个文件")
            return result
            
        except Exception as e:
            logger.error(f"分析目录失败: {str(e)}")
            return {}
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """分析单个文件"""
        try:
            # 获取文件信息
            file_info = self.processor.get_document_info(file_path)
            
            if not file_info.get('supported', False):
                return None
            
            # 提取文本内容
            try:
                documents = self.processor.process_document(file_path)
                if not documents:
                    return None
                
                # 合并所有文档块的内容
                content = '\n'.join([doc.page_content for doc in documents])
                
            except Exception as e:
                logger.warning(f"处理文档失败 {file_path}: {str(e)}")
                return None
            
            # 文本分析
            text_stats = self.analyze_text(content)
            
            # 质量评分
            quality_score = self.calculate_quality_score(content, text_stats)
            
            stats = {
                'file_path': file_path,
                'filename': file_info['filename'],
                'file_type': file_info['extension'],
                'size_bytes': file_info['size'],
                'size_mb': file_info['size_mb'],
                'content_length': len(content),
                'word_count': text_stats['word_count'],
                'sentence_count': text_stats['sentence_count'],
                'paragraph_count': text_stats['paragraph_count'],
                'avg_sentence_length': text_stats['avg_sentence_length'],
                'avg_word_length': text_stats['avg_word_length'],
                'quality_score': quality_score,
                'language_complexity': text_stats['language_complexity'],
                'information_density': text_stats['information_density']
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"分析文件失败 {file_path}: {str(e)}")
            return None
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """分析文本内容"""
        try:
            # 基本统计
            sentences = re.split(r'[。！？.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            paragraphs = text.split('\n\n')
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            
            # 分词
            words = list(jieba.cut(text))
            words = [w.strip() for w in words if w.strip() and len(w) > 1]
            
            # 计算统计量
            word_count = len(words)
            sentence_count = len(sentences)
            paragraph_count = len(paragraphs)
            
            avg_sentence_length = sum(len(s) for s in sentences) / sentence_count if sentence_count > 0 else 0
            avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0
            
            # 语言复杂度（词汇丰富度）
            unique_words = len(set(words))
            language_complexity = unique_words / word_count if word_count > 0 else 0
            
            # 信息密度（数字、专业术语的比例）
            numbers = re.findall(r'\d+', text)
            english_words = re.findall(r'[a-zA-Z]+', text)
            special_terms = re.findall(r'[A-Z]{2,}', text)  # 大写缩写
            
            information_density = (len(numbers) + len(english_words) + len(special_terms)) / word_count if word_count > 0 else 0
            
            return {
                'word_count': word_count,
                'sentence_count': sentence_count,
                'paragraph_count': paragraph_count,
                'avg_sentence_length': avg_sentence_length,
                'avg_word_length': avg_word_length,
                'language_complexity': language_complexity,
                'information_density': information_density,
                'unique_words': unique_words,
                'numbers_count': len(numbers),
                'english_words_count': len(english_words)
            }
            
        except Exception as e:
            logger.error(f"文本分析失败: {str(e)}")
            return {}
    
    def calculate_quality_score(self, content: str, text_stats: Dict[str, Any]) -> float:
        """计算文档质量得分"""
        try:
            score = 0.0
            
            # 长度得分（0-30分）
            length = len(content)
            if length < 100:
                length_score = 0
            elif length < 500:
                length_score = 10
            elif length < 2000:
                length_score = 20
            elif length < 10000:
                length_score = 30
            else:
                length_score = 25  # 太长的文档可能质量不高
            
            score += length_score
            
            # 结构得分（0-25分）
            paragraph_count = text_stats.get('paragraph_count', 0)
            sentence_count = text_stats.get('sentence_count', 0)
            
            if paragraph_count > 1 and sentence_count > 5:
                structure_score = 25
            elif paragraph_count > 0 and sentence_count > 2:
                structure_score = 15
            else:
                structure_score = 5
            
            score += structure_score
            
            # 语言复杂度得分（0-20分）
            complexity = text_stats.get('language_complexity', 0)
            if complexity > 0.7:
                complexity_score = 20
            elif complexity > 0.5:
                complexity_score = 15
            elif complexity > 0.3:
                complexity_score = 10
            else:
                complexity_score = 5
            
            score += complexity_score
            
            # 信息密度得分（0-15分）
            density = text_stats.get('information_density', 0)
            if density > 0.1:
                density_score = 15
            elif density > 0.05:
                density_score = 10
            elif density > 0.02:
                density_score = 5
            else:
                density_score = 0
            
            score += density_score
            
            # 平均句长得分（0-10分）
            avg_sentence_length = text_stats.get('avg_sentence_length', 0)
            if 20 <= avg_sentence_length <= 100:
                sentence_score = 10
            elif 10 <= avg_sentence_length <= 150:
                sentence_score = 7
            else:
                sentence_score = 3
            
            score += sentence_score
            
            # 标准化到0-1范围
            return min(score / 100, 1.0)
            
        except Exception as e:
            logger.error(f"质量评分失败: {str(e)}")
            return 0.0
    
    def generate_report(self, output_path: str = None):
        """生成分析报告"""
        try:
            if not self.stats:
                logger.error("没有统计数据，请先运行分析")
                return
            
            report = self._create_text_report()
            
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                logger.info(f"报告已保存到: {output_path}")
            else:
                print(report)
                
        except Exception as e:
            logger.error(f"生成报告失败: {str(e)}")
    
    def _create_text_report(self) -> str:
        """创建文本报告"""
        stats = self.stats
        
        report = f"""
# 数据集质量分析报告

## 总体统计

- 总文件数: {stats['overall_stats']['total_files']}
- 总大小: {stats['overall_stats']['total_size_mb']:.2f} MB
- 总词数: {stats['overall_stats']['total_words']:,}
- 平均文件大小: {stats['overall_stats']['average_file_size_kb']:.2f} KB
- 平均每文件词数: {stats['overall_stats']['average_words_per_file']:.0f}

## 文件类型分布

"""
        
        for file_type, count in stats['file_types'].items():
            percentage = (count / stats['overall_stats']['total_files']) * 100
            report += f"- {file_type}: {count} 个文件 ({percentage:.1f}%)\n"
        
        report += f"""
## 质量分布

- 平均质量得分: {stats['quality_distribution']['mean']:.3f}
- 标准差: {stats['quality_distribution']['std']:.3f}
- 最低得分: {stats['quality_distribution']['min']:.3f}
- 最高得分: {stats['quality_distribution']['max']:.3f}

## 质量建议

"""
        
        mean_quality = stats['quality_distribution']['mean']
        if mean_quality >= 0.8:
            report += "✅ 数据集质量优秀，可以直接用于RAG系统\n"
        elif mean_quality >= 0.6:
            report += "⚠️ 数据集质量良好，建议进行少量清洗\n"
        elif mean_quality >= 0.4:
            report += "⚠️ 数据集质量一般，需要进行质量提升\n"
        else:
            report += "❌ 数据集质量较差，建议重新收集或大幅改进\n"
        
        report += f"""
## 改进建议

1. **数据规模**: 当前有{stats['overall_stats']['total_files']}个文件，建议增加到1000+个文件以提高RAG效果
2. **文档长度**: 确保每个文档包含足够的信息（建议500字以上）
3. **内容质量**: 提高文档的信息密度和专业性
4. **格式统一**: 统一文档格式和结构
5. **去重处理**: 检查并移除重复或相似的内容

## 详细统计

### 高质量文档 (得分 > 0.8)
"""
        
        high_quality_files = [f for f in stats['file_stats'] if f['quality_score'] > 0.8]
        report += f"共 {len(high_quality_files)} 个文件\n"
        
        for file_info in high_quality_files[:5]:  # 显示前5个
            report += f"- {file_info['filename']}: {file_info['quality_score']:.3f}\n"
        
        report += f"""
### 低质量文档 (得分 < 0.4)
"""
        
        low_quality_files = [f for f in stats['file_stats'] if f['quality_score'] < 0.4]
        report += f"共 {len(low_quality_files)} 个文件\n"
        
        for file_info in low_quality_files[:5]:  # 显示前5个
            report += f"- {file_info['filename']}: {file_info['quality_score']:.3f}\n"
        
        return report
    
    def create_visualizations(self, output_dir: str = "./analysis_charts"):
        """创建可视化图表"""
        try:
            if not self.stats:
                logger.error("没有统计数据，请先运行分析")
                return
            
            os.makedirs(output_dir, exist_ok=True)
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 1. 文件类型分布饼图
            plt.figure(figsize=(10, 6))
            file_types = self.stats['file_types']
            plt.pie(file_types.values(), labels=file_types.keys(), autopct='%1.1f%%')
            plt.title('文件类型分布')
            plt.savefig(os.path.join(output_dir, '文件类型分布.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 质量得分分布直方图
            plt.figure(figsize=(10, 6))
            quality_scores = [f['quality_score'] for f in self.stats['file_stats']]
            plt.hist(quality_scores, bins=20, alpha=0.7, color='skyblue')
            plt.xlabel('质量得分')
            plt.ylabel('文件数量')
            plt.title('质量得分分布')
            plt.axvline(np.mean(quality_scores), color='red', linestyle='--', label=f'平均值: {np.mean(quality_scores):.3f}')
            plt.legend()
            plt.savefig(os.path.join(output_dir, '质量得分分布.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. 文件大小分布
            plt.figure(figsize=(10, 6))
            file_sizes = [f['size_mb'] for f in self.stats['file_stats']]
            plt.hist(file_sizes, bins=20, alpha=0.7, color='lightgreen')
            plt.xlabel('文件大小 (MB)')
            plt.ylabel('文件数量')
            plt.title('文件大小分布')
            plt.savefig(os.path.join(output_dir, '文件大小分布.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 4. 词数分布
            plt.figure(figsize=(10, 6))
            word_counts = [f['word_count'] for f in self.stats['file_stats']]
            plt.hist(word_counts, bins=20, alpha=0.7, color='orange')
            plt.xlabel('词数')
            plt.ylabel('文件数量')
            plt.title('文档词数分布')
            plt.savefig(os.path.join(output_dir, '词数分布.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"图表已保存到: {output_dir}")
            
        except Exception as e:
            logger.error(f"创建可视化失败: {str(e)}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据集质量分析工具")
    parser.add_argument("directory", help="要分析的文档目录")
    parser.add_argument("--report", type=str, help="报告输出路径")
    parser.add_argument("--charts", type=str, help="图表输出目录")
    parser.add_argument("--json", type=str, help="JSON统计输出路径")
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = DatasetAnalyzer()
    
    # 分析目录
    print("开始分析数据集...")
    stats = analyzer.analyze_directory(args.directory)
    
    if not stats:
        print("分析失败")
        return
    
    # 生成报告
    print("生成分析报告...")
    analyzer.generate_report(args.report)
    
    # 创建图表
    if args.charts:
        print("创建可视化图表...")
        analyzer.create_visualizations(args.charts)
    
    # 保存JSON统计
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"JSON统计已保存到: {args.json}")
    
    print("分析完成！")

if __name__ == "__main__":
    main()
