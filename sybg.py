"""
实验二：使用Deepseek-R1和V3构建RAG应用
功能：支持多模型选择、混合检索策略、可视化评估
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 配置类 ====================
class Config:
    """实验配置"""
    # API配置
    DEEPSEEK_API_KEY = "sk-8454f30b99cc4305a20ea976f5c3f9ac"  # 替换为实际API Key
    DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

    # 本地Ollama配置（可选）
    OLLAMA_URL = "http://localhost:11434/v1"

    # 模型选择
    AVAILABLE_MODELS = {
        "deepseek-r1": "deepseek-reasoner",  # 推理模型
        "deepseek-v3": "deepseek-chat",  # 生成模型
        "qwen2.5": "qwen2.5:7b",  # 本地模型
        "qwen2.5-0.5b": "qwen2.5:0.5b"
    }

    # 检索配置
    RETRIEVAL_TOP_K = 5
    SEMANTIC_WEIGHT = 0.7  # 语义检索权重
    KEYWORD_WEIGHT = 0.3  # 关键词检索权重

    # 文件路径
    KNOWLEDGE_FILE = "knowledge_base.txt"
    VECTOR_DB_PATH = "faiss_index"

    # 测试问题
    TEST_QUESTIONS = [
        {"type": "事实型", "question": "什么是RAG技术？", "expected_keywords": ["检索", "增强", "生成"]},
        {"type": "推理型", "question": "为什么RAG比纯大模型更适合知识密集型任务？",
         "expected_keywords": ["外部知识", "幻觉", "实时更新"]},
        {"type": "多跳推理型", "question": "如果用户问'Deepseek模型的特点是什么，它和RAG有什么关系？'",
         "expected_keywords": ["推理能力", "检索增强", "准确性"]}
    ]


# ==================== 数据预处理模块 ====================
class DataProcessor:
    """数据预处理与分块"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_knowledge_base(self, file_path: str) -> str:
        """加载知识库文件"""
        if not os.path.exists(file_path):
            # 创建示例知识库
            self._create_sample_knowledge(file_path)

        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _create_sample_knowledge(self, file_path: str):
        """创建示例知识库"""
        sample_content = """
# RAG技术介绍

RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合检索和生成的大语言模型技术。
它通过从外部知识库中检索相关信息，增强模型的生成能力。

## RAG的核心优势

1. 知识更新：可以实时更新知识库，无需重新训练模型
2. 可解释性：生成结果可追溯到检索来源
3. 减少幻觉：基于事实检索，降低模型虚构内容的风险
4. 领域适应：快速适配特定领域知识

## Deepseek模型介绍

Deepseek系列模型包括：
- Deepseek-V3：高性能通用大语言模型，擅长文本生成和对话
- Deepseek-R1：专注推理的模型，具有更强的逻辑推理能力

## RAG工作流程

1. 文档加载与分块
2. 文本向量化（Embedding）
3. 向量存储与索引（FAISS）
4. 检索相关文档
5. 结合检索结果生成回答

## 混合检索策略

混合检索结合语义检索和关键词检索：
- 语义检索：基于向量相似度，理解语义含义
- 关键词检索：基于BM25算法，匹配精确关键词
- 权重配比：语义70% + 关键词30%

## 评估指标

- 准确率：回答与参考答案的匹配程度
- 召回率：检索到的相关文档比例
- 响应时间：从提问到回答的总耗时
- 答案完整性：是否覆盖问题所有方面
"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(sample_content)
        print(f"✅ 已创建示例知识库: {file_path}")

    def split_text(self, text: str) -> List[str]:
        """文本分块"""
        chunks = []
        words = text.split('\n')
        current_chunk = []
        current_length = 0

        for word in words:
            if current_length + len(word) <= self.chunk_size:
                current_chunk.append(word)
                current_length += len(word)
            else:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                # 重叠部分
                overlap_size = self.chunk_overlap
                current_chunk = current_chunk[-overlap_size:] if overlap_size > 0 else []
                current_chunk.append(word)
                current_length = sum(len(w) for w in current_chunk)

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        print(f"✅ 文本分块完成，共 {len(chunks)} 个块")
        return chunks


# ==================== 向量化与检索模块 ====================
class VectorRetriever:
    """向量检索与混合检索"""

    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.documents = []
        self.bm25_index = None

    def init_embeddings(self):
        """初始化Embedding模型"""
        try:
            from sentence_transformers import SentenceTransformer
            import os

            # 检查本地模型是否存在
            local_model_path = "models/AI-ModelScope/bge-large-zh-v1_5"

            if os.path.exists(local_model_path):
                print(f"使用本地模型: {local_model_path}")
                self.embeddings = SentenceTransformer(local_model_path)
            else:
                # 如果没有本地模型，尝试从ModelScope下载
                print("本地模型不存在，正在从ModelScope下载...")
                from modelscope.hub.snapshot_download import snapshot_download
                snapshot_download('AI-ModelScope/bge-large-zh-v1.5', cache_dir='models')
                self.embeddings = SentenceTransformer(local_model_path)

            print("✅ Embedding模型加载完成")
            return True
        except Exception as e:
            print(f"❌ Embedding模型加载失败: {e}")
            return False

    def build_index(self, documents: List[str]):
        """构建向量索引"""
        try:
            import faiss
            self.documents = documents

            # 生成向量
            print("正在生成文档向量...")
            vectors = self.embeddings.encode(documents, show_progress_bar=True)
            vectors = np.array(vectors).astype('float32')

            # 构建FAISS索引
            dimension = vectors.shape[1]
            self.vector_store = faiss.IndexFlatL2(dimension)
            self.vector_store.add(vectors)

            print(f"✅ 向量索引构建完成，维度: {dimension}")
            return True
        except Exception as e:
            print(f"❌ 向量索引构建失败: {e}")
            return False

    def build_bm25_index(self):
        """构建BM25关键词索引"""
        try:
            from rank_bm25 import BM25Okapi
            tokenized_docs = [doc.split() for doc in self.documents]
            self.bm25_index = BM25Okapi(tokenized_docs)
            print("✅ BM25索引构建完成")
            return True
        except Exception as e:
            print(f"⚠️ BM25索引构建失败: {e}，将仅使用语义检索")
            return False

    def semantic_search(self, query: str, top_k: int = Config.RETRIEVAL_TOP_K) -> List[Tuple[int, float]]:
        """语义检索"""
        if self.vector_store is None:
            return []

        query_vector = self.embeddings.encode([query]).astype('float32')
        distances, indices = self.vector_store.search(query_vector, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append((idx, float(distances[0][i])))
        return results

    def keyword_search(self, query: str, top_k: int = Config.RETRIEVAL_TOP_K) -> List[Tuple[int, float]]:
        """关键词检索（BM25）"""
        if self.bm25_index is None:
            return []

        tokenized_query = query.split()
        scores = self.bm25_index.get_scores(tokenized_query)

        # 获取top_k结果
        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = [(idx, float(scores[idx])) for idx in top_indices if scores[idx] > 0]
        return results

    def hybrid_search(self, query: str, semantic_weight: float = Config.SEMANTIC_WEIGHT,
                      keyword_weight: float = Config.KEYWORD_WEIGHT,
                      top_k: int = Config.RETRIEVAL_TOP_K) -> List[Tuple[int, float, Dict]]:
        """混合检索"""
        # 执行两种检索
        semantic_results = self.semantic_search(query, top_k)
        keyword_results = self.keyword_search(query, top_k)

        # 归一化分数
        semantic_scores = {idx: score for idx, score in semantic_results}
        keyword_scores = {idx: score for idx, score in keyword_results}

        # 归一化
        if semantic_scores:
            max_semantic = max(semantic_scores.values())
            semantic_scores = {k: v / max_semantic for k, v in semantic_scores.items()}

        if keyword_scores:
            max_keyword = max(keyword_scores.values())
            keyword_scores = {k: v / max_keyword for k, v in keyword_scores.items()}

        # 合并分数
        all_indices = set(semantic_scores.keys()) | set(keyword_scores.keys())
        hybrid_scores = {}

        for idx in all_indices:
            sem_score = semantic_scores.get(idx, 0) * semantic_weight
            kw_score = keyword_scores.get(idx, 0) * keyword_weight
            hybrid_scores[idx] = sem_score + kw_score

        # 排序并返回
        sorted_results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in sorted_results:
            results.append((idx, score, {
                "semantic_score": semantic_scores.get(idx, 0),
                "keyword_score": keyword_scores.get(idx, 0),
                "content": self.documents[idx][:200] + "..."
            }))

        return results

    def get_document(self, idx: int) -> str:
        """获取文档内容"""
        return self.documents[idx] if 0 <= idx < len(self.documents) else ""


# ==================== 模型调用模块 ====================
class ModelClient:
    """Deepseek模型客户端"""

    def __init__(self, model_type: str = "deepseek-v3"):
        self.model_type = model_type
        self.client = None
        self.model_name = Config.AVAILABLE_MODELS.get(model_type, Config.AVAILABLE_MODELS["deepseek-v3"])
        self.use_local = False

    def init_client(self):
        """初始化客户端"""
        try:
            from openai import OpenAI

            # 尝试使用Deepseek API
            if Config.DEEPSEEK_API_KEY and Config.DEEPSEEK_API_KEY != "your-deepseek-api-key":
                self.client = OpenAI(
                    api_key=Config.DEEPSEEK_API_KEY,
                    base_url=Config.DEEPSEEK_BASE_URL
                )
                print(f"✅ Deepseek API连接成功，模型: {self.model_name}")
            else:
                # 使用本地Ollama
                self.client = OpenAI(
                    api_key="ollama",
                    base_url=Config.OLLAMA_URL
                )
                self.use_local = True
                print(f"✅ 本地Ollama连接成功，模型: {self.model_name}")
            return True
        except Exception as e:
            print(f"❌ 模型客户端初始化失败: {e}")
            return False

    def generate(self, prompt: str, system_prompt: str = "你是一个专业的AI助手。",
                 max_tokens: int = 512, temperature: float = 0.7) -> str:
        """生成回答"""
        if self.client is None:
            if not self.init_client():
                return "模型连接失败"

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

            # 如果是Deepseek-R1，使用推理模式
            if self.model_type == "deepseek-r1" and not self.use_local:
                messages.insert(0, {"role": "system", "content": "请逐步推理，给出详细思考过程。"})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            return response.choices[0].message.content
        except Exception as e:
            return f"生成失败: {e}"

    def generate_with_context(self, query: str, context: str, system_prompt: str = None) -> str:
        """基于上下文生成回答"""
        if system_prompt is None:
            system_prompt = """你是一个专业的AI助手。请基于提供的上下文信息回答用户问题。
如果上下文信息不足以回答问题，请说明这一点，不要编造答案。
回答要准确、简洁、有条理。"""

        prompt = f"""上下文信息：
{context}

用户问题：{query}

请基于以上上下文信息回答用户问题："""

        return self.generate(prompt, system_prompt)


# ==================== RAG应用主类 ====================
class RAGApplication:
    """RAG应用主类"""

    def __init__(self, model_type: str = "deepseek-v3"):
        self.model_type = model_type
        self.data_processor = DataProcessor()
        self.retriever = VectorRetriever()
        self.model_client = ModelClient(model_type)
        self.is_initialized = False
        self.performance_metrics = []

    def initialize(self, knowledge_file: str = Config.KNOWLEDGE_FILE):
        """初始化RAG系统"""
        print("\n" + "=" * 50)
        print("初始化RAG系统...")
        print("=" * 50)

        # 1. 加载知识库
        print("\n[1/4] 加载知识库...")
        text = self.data_processor.load_knowledge_base(knowledge_file)

        # 2. 文本分块
        print("\n[2/4] 文本分块...")
        chunks = self.data_processor.split_text(text)

        # 3. 构建向量索引
        print("\n[3/4] 构建向量索引...")
        if not self.retriever.init_embeddings():
            return False
        if not self.retriever.build_index(chunks):
            return False

        # 4. 构建BM25索引
        print("\n[4/4] 构建关键词索引...")
        self.retriever.build_bm25_index()

        # 5. 初始化模型
        print("\n[5/5] 初始化模型...")
        if not self.model_client.init_client():
            return False

        self.is_initialized = True
        print("\n✅ RAG系统初始化完成！")
        return True

    def query(self, question: str, return_details: bool = False) -> Dict:
        """执行RAG查询"""
        if not self.is_initialized:
            return {"error": "系统未初始化"}

        start_time = time.time()

        # 1. 检索相关文档
        retrieve_start = time.time()
        search_results = self.retriever.hybrid_search(question)
        retrieve_time = time.time() - retrieve_start

        # 2. 构建上下文
        context = "\n\n".join([self.retriever.get_document(idx) for idx, _, _ in search_results])

        # 3. 生成回答
        generate_start = time.time()
        answer = self.model_client.generate_with_context(question, context)
        generate_time = time.time() - generate_start

        total_time = time.time() - start_time

        # 4. 记录指标
        metrics = {
            "question": question,
            "model": self.model_type,
            "retrieve_time": retrieve_time,
            "generate_time": generate_time,
            "total_time": total_time,
            "retrieved_count": len(search_results),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.performance_metrics.append(metrics)

        result = {
            "answer": answer,
            "metrics": metrics
        }

        if return_details:
            result["context"] = context
            result["retrieved_docs"] = [
                {"index": idx, "score": score, "details": details}
                for idx, score, details in search_results
            ]

        return result

    def compare_without_rag(self, question: str) -> Dict:
        """对比：不使用RAG的直接回答"""
        start_time = time.time()
        answer = self.model_client.generate(question)
        total_time = time.time() - start_time

        return {
            "answer": answer,
            "metrics": {
                "total_time": total_time,
                "model": self.model_type,
                "with_rag": False
            }
        }


# ==================== 实验评估模块 ====================
class ExperimentEvaluator:
    """实验评估与可视化"""

    def __init__(self, rag_app: RAGApplication):
        self.rag_app = rag_app
        self.results = []

    def run_tests(self):
        """运行测试"""
        print("\n" + "=" * 50)
        print("开始实验测试...")
        print("=" * 50)

        for test in Config.TEST_QUESTIONS:
            print(f"\n📝 测试问题 [{test['type']}]: {test['question']}")

            # RAG方案
            print("  🔍 RAG方案回答中...")
            rag_result = self.rag_app.query(test['question'], return_details=False)

            # 单独模型方案
            print("  🤖 单独模型回答中...")
            alone_result = self.rag_app.compare_without_rag(test['question'])

            self.results.append({
                "type": test['type'],
                "question": test['question'],
                "rag_answer": rag_result['answer'],
                "alone_answer": alone_result['answer'],
                "rag_metrics": rag_result['metrics'],
                "alone_metrics": alone_result['metrics'],
                "expected_keywords": test.get('expected_keywords', [])
            })

            print(f"  ✅ RAG耗时: {rag_result['metrics']['total_time']:.2f}s")
            print(f"  ✅ 单独模型耗时: {alone_result['metrics']['total_time']:.2f}s")

    def evaluate_accuracy(self, answer: str, expected_keywords: List[str]) -> float:
        """评估准确率（基于关键词匹配）"""
        if not expected_keywords:
            return 0.5
        matched = sum(1 for kw in expected_keywords if kw in answer)
        return matched / len(expected_keywords)

    def generate_report(self):
        """生成实验报告"""
        print("\n" + "=" * 60)
        print("实验报告")
        print("=" * 60)

        # 创建DataFrame
        data = []
        for r in self.results:
            rag_acc = self.evaluate_accuracy(r['rag_answer'], r['expected_keywords'])
            alone_acc = self.evaluate_accuracy(r['alone_answer'], r['expected_keywords'])

            data.append({
                "问题类型": r['type'],
                "RAG回答时间(s)": f"{r['rag_metrics']['total_time']:.2f}",
                "单独模型时间(s)": f"{r['alone_metrics']['total_time']:.2f}",
                "RAG准确率": f"{rag_acc:.1%}",
                "单独模型准确率": f"{alone_acc:.1%}",
                "检索文档数": r['rag_metrics']['retrieved_count']
            })

        df = pd.DataFrame(data)
        print("\n📊 性能对比表:")
        print(df.to_string(index=False))

        # 计算平均值
        avg_rag_time = np.mean([r['rag_metrics']['total_time'] for r in self.results])
        avg_alone_time = np.mean([r['alone_metrics']['total_time'] for r in self.results])

        print(f"\n📈 平均响应时间对比:")
        print(f"   RAG方案: {avg_rag_time:.2f}秒")
        print(f"   单独模型: {avg_alone_time:.2f}秒")
        print(f"   时间增加: {(avg_rag_time - avg_alone_time):.2f}秒")

        return df

    def visualize(self):
        """可视化结果"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 响应时间对比
        types = [r['type'] for r in self.results]
        rag_times = [r['rag_metrics']['total_time'] for r in self.results]
        alone_times = [r['alone_metrics']['total_time'] for r in self.results]

        x = np.arange(len(types))
        width = 0.35
        ax1 = axes[0, 0]
        bars1 = ax1.bar(x - width / 2, rag_times, width, label='RAG方案', color='#2E86AB')
        bars2 = ax1.bar(x + width / 2, alone_times, width, label='单独模型', color='#A23B72')
        ax1.set_xlabel('问题类型')
        ax1.set_ylabel('响应时间 (秒)')
        ax1.set_title('响应时间对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(types)
        ax1.legend()

        # 添加数值标签
        for bar in bars1:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f'{bar.get_height():.1f}s', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f'{bar.get_height():.1f}s', ha='center', va='bottom', fontsize=9)

        # 2. 准确率对比
        ax2 = axes[0, 1]
        rag_acc = [self.evaluate_accuracy(r['rag_answer'], r['expected_keywords']) for r in self.results]
        alone_acc = [self.evaluate_accuracy(r['alone_answer'], r['expected_keywords']) for r in self.results]

        bars3 = ax2.bar(x - width / 2, rag_acc, width, label='RAG方案', color='#2E86AB')
        bars4 = ax2.bar(x + width / 2, alone_acc, width, label='单独模型', color='#A23B72')
        ax2.set_xlabel('问题类型')
        ax2.set_ylabel('准确率')
        ax2.set_title('回答准确率对比')
        ax2.set_xticks(x)
        ax2.set_xticklabels(types)
        ax2.set_ylim(0, 1.1)
        ax2.legend()

        for bar in bars3:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{bar.get_height():.0%}', ha='center', va='bottom', fontsize=9)
        for bar in bars4:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{bar.get_height():.0%}', ha='center', va='bottom', fontsize=9)

        # 3. 检索文档数量
        ax3 = axes[1, 0]
        retrieved_counts = [r['rag_metrics']['retrieved_count'] for r in self.results]
        ax3.bar(types, retrieved_counts, color='#F18F01')
        ax3.set_xlabel('问题类型')
        ax3.set_ylabel('检索文档数')
        ax3.set_title('各问题检索文档数量')
        for i, v in enumerate(retrieved_counts):
            ax3.text(i, v + 0.1, str(v), ha='center', va='bottom')

        # 4. 时间分解
        ax4 = axes[1, 1]
        retrieve_times = [r['rag_metrics']['retrieve_time'] for r in self.results]
        generate_times = [r['rag_metrics']['generate_time'] for r in self.results]

        ax4.bar(types, retrieve_times, label='检索时间', color='#73AB84')
        ax4.bar(types, generate_times, bottom=retrieve_times, label='生成时间', color='#F4D58C')
        ax4.set_xlabel('问题类型')
        ax4.set_ylabel('时间 (秒)')
        ax4.set_title('RAG方案时间分解')
        ax4.legend()

        plt.tight_layout()

        # 保存图片
        plt.savefig('experiment_results.png', dpi=150, bbox_inches='tight')
        print("\n📊 可视化图表已保存: experiment_results.png")
        plt.show()

        return fig


# ==================== 交互式界面 ====================
class InteractiveUI:
    """交互式界面"""

    def __init__(self):
        self.rag_app = None

    def run(self):
        """运行交互界面"""
        print("\n" + "=" * 60)
        print("Deepseek-R1/V3 RAG应用实验系统")
        print("=" * 60)

        # 选择模型
        print("\n请选择模型:")
        print("1. Deepseek-V3 (推荐，生成能力强)")
        print("2. Deepseek-R1 (推理能力强)")
        print("3. Qwen2.5-7B (本地模型)")
        print("4. Qwen2.5-0.5B (轻量级)")

        model_choice = input("\n请输入选项 (1-4): ").strip()
        model_map = {"1": "deepseek-v3", "2": "deepseek-r1", "3": "qwen2.5", "4": "qwen2.5-0.5b"}
        model_type = model_map.get(model_choice, "deepseek-v3")

        # 初始化RAG应用
        self.rag_app = RAGApplication(model_type)
        if not self.rag_app.initialize():
            print("❌ RAG系统初始化失败")
            return

        while True:
            print("\n" + "-" * 40)
            print("请选择操作:")
            print("1. 提问 (RAG模式)")
            print("2. 运行完整实验评估")
            print("3. 查看性能指标")
            print("4. 退出")

            choice = input("\n请输入选项 (1-4): ").strip()

            if choice == "1":
                self._ask_question()
            elif choice == "2":
                self._run_experiment()
            elif choice == "3":
                self._show_metrics()
            elif choice == "4":
                print("再见！")
                break
            else:
                print("无效选项")

    def _ask_question(self):
        """提问模式"""
        question = input("\n请输入您的问题: ").strip()
        if not question:
            return

        print("\n🤔 思考中...")
        result = self.rag_app.query(question, return_details=True)

        print(f"\n📝 回答:\n{result['answer']}")
        print(f"\n⏱️ 耗时: {result['metrics']['total_time']:.2f}秒")
        print(f"📚 检索文档数: {result['metrics']['retrieved_count']}")

    def _run_experiment(self):
        """运行实验评估"""
        evaluator = ExperimentEvaluator(self.rag_app)
        evaluator.run_tests()
        evaluator.generate_report()
        evaluator.visualize()

    def _show_metrics(self):
        """显示性能指标"""
        if not self.rag_app.performance_metrics:
            print("暂无数据，请先进行提问或运行实验")
            return

        print("\n📊 历史性能指标:")
        for i, m in enumerate(self.rag_app.performance_metrics[-10:], 1):
            print(f"{i}. [{m['timestamp']}] 问题: {m['question'][:30]}...")
            print(f"   检索时间: {m['retrieve_time']:.2f}s, 生成时间: {m['generate_time']:.2f}s")
            print(f"   总耗时: {m['total_time']:.2f}s, 检索文档: {m['retrieved_count']}个")


# ==================== 主程序 ====================
def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║     实验二：使用Deepseek-R1和V3构建RAG应用                        ║
║     功能：混合检索 | 多模型支持 | 可视化评估                       ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    # 检查依赖
    try:
        import sentence_transformers
        import faiss
        import rank_bm25
        import pandas
        import matplotlib
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("\n请安装依赖:")
        print("pip install sentence-transformers faiss-cpu rank-bm25 pandas matplotlib")
        return

    # 启动交互界面
    ui = InteractiveUI()
    ui.run()


if __name__ == "__main__":
    main()