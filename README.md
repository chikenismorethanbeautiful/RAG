# 基于 Qwen2.5 的 RAG+Agent 完整项目

这是一个集成了基础推理、RAG 问答、Agent 智能代理、Web 服务和性能测试的完整 AI 应用项目，基于 Qwen2.5-0.5B-Instruct 模型构建，支持本地推理和 Ollama API 两种运行方式。

## 🌟 功能特性

- **基础推理**：基于 Qwen2.5 模型的对话交互，支持本地加载模型和 Ollama API 两种模式
- **RAG 问答**：结合向量数据库的检索增强生成，支持自定义文档问答
- **Agent 代理**：集成工具调用能力（时间查询、计算器、天气查询）
- **Web 服务**：FastAPI 后端 + Streamlit 前端，提供可视化聊天界面
- **性能测试**：对比本地 Transformers 和 Ollama API 的推理性能

## 📋 环境要求

- Python 3.8+
- 可选：CUDA 11.7+（GPU 加速）
- Ollama（可选，用于 API 模式）

## 🚀 快速开始

### 1. 安装依赖

bash

运行

```
# 基础依赖安装
pip install torch transformers langchain langchain-openai langchain-community langchain-huggingface fastapi uvicorn streamlit openai sentence-transformers faiss-cpu

# 如果需要通过modelscope下载模型
pip install modelscope
```

### 2. 准备模型

#### 方式 1：自动下载（推荐）

运行项目并选择选项 1，自动下载模型到`models`目录：

bash

运行

```
python sanshinian.py
# 然后选择 1. 下载模型
```

#### 方式 2：手动下载

- Qwen2.5-0.5B-Instruct: https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct
- bge-large-zh-v1.5: https://www.modelscope.cn/models/AI-ModelScope/bge-large-zh-v1.5

下载后放入`models`目录，保持如下结构：

plaintext

```
models/
├── Qwen/
│   └── Qwen2.5-0.5B-Instruct/
└── AI-ModelScope/
    └── bge-large-zh-v1.5/
```

#### 方式 3：使用 Ollama（推荐轻量化部署）

bash

运行

```
# 启动Ollama服务
ollama serve

# 拉取Qwen2.5模型
ollama pull qwen2.5:0.5b
```

### 3. 运行项目

bash

运行

python sanshinian.py

根据提示选择功能：

```
1. 下载模型
2. 基础推理演示
3. RAG问答演示
4. Agent代理演示
5. 启动Web服务（FastAPI + Streamlit）
6. 性能测试
7. 退出
```

## 📖 功能使用说明

### 1. 基础推理演示

- 支持本地模型推理和 Ollama API 两种方式
- 输入`quit`/`exit`/`q`退出对话
- 实时对话交互，支持自定义系统提示词

### 2. RAG 问答演示

1. 选择选项 3 进入 RAG 模式
2. 输入本地 txt 文档路径（支持 UTF-8 编码）
3. 系统自动分割文档、创建向量索引
4. 输入问题进行基于文档的问答
5. 输入`quit`退出

### 3. Agent 代理演示

集成的工具：

- **当前时间**：获取系统当前时间
- **计算器**：支持数学表达式计算（abs/round/max/min 等函数）
- **天气查询**：需输入心知天气 API Key（https://www.seniverse.com/）

使用示例：

plaintext

```
问题: 现在几点了？
问题: 计算 2 + 3 * 4
问题: 北京今天天气怎么样？
```

### 4. Web 服务

选择选项 5 自动启动：

- FastAPI 后端：http://localhost:6066
- Streamlit 前端：http://localhost:8501

前端功能：

- 自定义系统提示词
- 调整推理参数（temperature/top_p/max_tokens）
- 流式 / 非流式输出切换
- 聊天历史管理

### 5. 性能测试

对比两种推理方式的性能：

- **Transformers 本地推理**：直接加载模型到 GPU/CPU
- **Ollama API**：通过 API 调用 Ollama 服务

测试指标：生成 token 数、耗时、tokens / 秒

## ⚙️ 配置说明

可修改代码中的`Config`类调整配置：

python

运行

```
class Config:
    # 模型配置
    MODEL_NAME = "qwen2.5:0.5b"  # Ollama模型名
    MODEL_PATH = "models/Qwen/Qwen2.5-0.5B-Instruct"  # 本地模型路径

    # API配置
    OLLAMA_URL = "http://localhost:11434/v1"
    API_KEY = "ollama"

    # 服务配置
    BACKEND_PORT = 6066  # FastAPI端口
    STREAMLIT_PORT = 8501  # Streamlit端口

    # RAG配置
    CHUNK_SIZE = 200  # 文档分块大小
    CHUNK_OVERLAP = 20  # 分块重叠长度
    EMBEDDING_MODEL = "models/AI-ModelScope/bge-large-zh-v1_5"  # 嵌入模型路径
```

## 📝 注意事项

1. 首次运行需确保模型下载完成，网络不佳时建议手动下载
2. 使用 Ollama 模式前需确保 Ollama 服务已启动：`ollama serve`
3. GPU 环境下自动使用 CUDA 加速，CPU 环境自动降级为 float32
4. RAG 功能仅支持 txt 格式文档，其他格式需扩展`load_document`方法
5. 天气查询功能需要心知天气 API Key，可免费申请

## 🐛 常见问题

### Q1: Ollama 连接失败

A1: 确保 Ollama 已启动：

bash

运行

```
ollama serve  # 启动服务
ollama list   # 检查模型是否已下载
```

### Q2: 模型加载内存不足

A2:

- CPU 运行时降低`max_tokens`参数
- GPU 运行时确保有足够显存（至少 4GB）
- 使用更小的模型版本

### Q3: RAG 问答返回空

A3:

- 检查文档编码是否为 UTF-8
- 调整`CHUNK_SIZE`和`CHUNK_OVERLAP`参数
- 确认嵌入模型路径正确

## 📄 许可证

本项目仅供学习使用，模型使用请遵循 Qwen2.5 的开源许可证。

# 🐕使用展示

![zy](.\zy.png)