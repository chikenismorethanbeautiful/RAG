"""
完整RAG+Agent项目 - 基于Qwen2.5
运行方式：
1. 确保ollama已启动: ollama run qwen2.5:0.5b
2. 运行本文件: python app.py
3. 选择功能:
   - 选项1: 基础推理
   - 选项2: 启动Web服务（FastAPI + Streamlit）
   - 选项3: RAG问答
   - 选项4: Agent代理
   - 选项5: 性能测试
"""

import os
import sys
import json
import time
import requests
import threading
import webbrowser
from typing import List, Dict, Optional
from datetime import datetime

# ==================== 检查依赖 ====================
def check_dependencies():
    """检查并提示安装缺失的依赖"""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    try:
        import langchain
    except ImportError:
        missing.append("langchain")
    try:
        import fastapi
    except ImportError:
        missing.append("fastapi")
    try:
        import streamlit
    except ImportError:
        missing.append("streamlit")
    try:
        import openai
    except ImportError:
        missing.append("openai")

    if missing:
        print("缺少以下依赖，请先安装:")
        print(f"pip install {' '.join(missing)}")
        print("\n完整安装命令:")
        print("pip install torch transformers langchain langchain-openai langchain-community langchain-huggingface fastapi uvicorn streamlit openai sentence-transformers faiss-cpu")
        return False
    return True

# ==================== 配置 ====================
class Config:
    """配置类"""
    # 模型配置
    MODEL_NAME = "qwen2.5:0.5b"  # ollama中的模型名
    MODEL_PATH = "models/Qwen/Qwen2.5-0.5B-Instruct"  # 本地模型路径

    # API配置
    OLLAMA_URL = "http://localhost:11434/v1"
    API_KEY = "ollama"

    # 服务配置
    BACKEND_PORT = 6066
    STREAMLIT_PORT = 8501

    # RAG配置
    CHUNK_SIZE = 200
    CHUNK_OVERLAP = 20
    EMBEDDING_MODEL = "models/AI-ModelScope/bge-large-zh-v1_5"

# ==================== 基础推理模块 ====================
# ==================== 基础推理模块（使用Ollama API）====================
class BasicInference:
    """基础推理类 - 使用Ollama API"""

    def __init__(self):
        self.client = None

    def init_client(self):
        """初始化Ollama客户端"""
        try:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=Config.API_KEY,
                base_url=Config.OLLAMA_URL
            )

            # 测试连接
            self.client.models.list()
            print("Ollama连接成功")
            return True
        except Exception as e:
            print(f"Ollama连接失败: {e}")
            print("请确保Ollama已启动: ollama serve")
            print("并且已下载模型: ollama pull qwen2.5:0.5b")
            return False

    def chat(self, prompt: str, system_prompt: str = "You are a helpful assistant.",
             max_tokens: int = 512) -> str:
        """对话"""
        if self.client is None:
            if not self.init_client():
                return "Ollama未连接"

        try:
            response = self.client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"调用失败: {e}"

    def run_demo(self):
        """运行演示"""
        if not self.init_client():
            return

        print("\n=== 基础推理演示（使用Ollama）===")
        print("输入 'quit' 退出")

        while True:
            user_input = input("\n你: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            print("助手: ", end="", flush=True)
            response = self.chat(user_input)
            print(response)

    def load_model(self):
        """加载模型"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"使用设备: {self.device}")

            print(f"加载模型: {Config.MODEL_PATH}")
            self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)
            self.model = AutoModelForCausalLM.from_pretrained(
                Config.MODEL_PATH,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to(self.device)
            print("模型加载完成")
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("请先运行 download_model.py 下载模型")
            return False

    def chat(self, prompt: str, system_prompt: str = "You are a helpful assistant.",
             max_new_tokens: int = 512) -> str:
        """对话"""
        if self.model is None:
            if not self.load_model():
                return "模型未加载"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7
        )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids
                        in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

    def run_demo(self):
        """运行演示"""
        if not self.load_model():
            return

        print("\n=== 基础推理演示 ===")
        print("输入 'quit' 退出")

        while True:
            user_input = input("\n你: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            print("助手: ", end="", flush=True)
            response = self.chat(user_input)
            print(response)

# ==================== RAG问答模块 ====================
class RAGQA:
    """RAG问答系统"""

    def __init__(self):
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        self.llm = None

    def init_llm(self):
        """初始化LLM"""
        try:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                openai_api_key=Config.API_KEY,
                base_url=Config.OLLAMA_URL,
                model=Config.MODEL_NAME
            )
            return True
        except Exception as e:
            print(f"LLM初始化失败: {e}")
            return False

    def create_vector_store(self, documents: List[str]):
        """创建向量存储"""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            # 文本分割
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )
            chunks = text_splitter.create_documents(documents)

            # 创建Embedding
            embedding = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)

            # 创建向量存储
            self.vector_store = FAISS.from_documents(chunks, embedding)
            self.retriever = self.vector_store.as_retriever()

            print(f"向量存储创建完成，共{len(chunks)}个文档块")
            return True
        except Exception as e:
            print(f"向量存储创建失败: {e}")
            return False

    def build_qa_chain(self):
        """构建问答链"""
        try:
            from langchain.chains import RetrievalQA
            from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

            system_message = SystemMessagePromptTemplate.from_template(
                "根据以下已知信息回答用户问题。如果不知道答案，就说不知道。\n已知信息{context}"
            )
            human_message = HumanMessagePromptTemplate.from_template("用户问题：{question}")
            chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                chain_type_kwargs={"prompt": chat_prompt}
            )
            return True
        except Exception as e:
            print(f"问答链创建失败: {e}")
            return False

    def load_document(self, file_path: str) -> List[str]:
        """加载文档"""
        try:
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            return [doc.page_content for doc in docs]
        except Exception as e:
            print(f"文档加载失败: {e}")
            return []

    def query(self, question: str) -> str:
        """查询"""
        if self.qa_chain is None:
            return "请先初始化RAG系统"

        try:
            result = self.qa_chain.invoke(question)
            return result.get("result", "")
        except Exception as e:
            return f"查询失败: {e}"

    def run_demo(self):
        """运行演示"""
        if not self.init_llm():
            return

        print("\n=== RAG问答演示 ===")
        file_path = input("请输入文档路径（支持txt文件）: ").strip()

        if not os.path.exists(file_path):
            print("文件不存在")
            return

        documents = self.load_document(file_path)
        if not documents:
            return

        if not self.create_vector_store(documents):
            return

        if not self.build_qa_chain():
            return

        print("RAG系统初始化完成，输入 'quit' 退出")

        while True:
            question = input("\n问题: ")
            if question.lower() in ['quit', 'exit', 'q']:
                break

            print("回答: ", end="", flush=True)
            answer = self.query(question)
            print(answer)

# ==================== Agent代理模块 ====================
class AgentDemo:
    """Agent代理"""

    def __init__(self):
        self.agent_executor = None
        self.tools = []

    def init_agent(self):
        """初始化Agent"""
        try:
            from langchain_openai import ChatOpenAI
            from langchain.agents import Tool, create_react_agent, AgentExecutor
            from langchain_core.prompts import PromptTemplate

            # 初始化LLM
            llm = ChatOpenAI(
                openai_api_key=Config.API_KEY,
                base_url=Config.OLLAMA_URL,
                model=Config.MODEL_NAME
            )

            # 定义工具
            tools = [
                Tool(name="当前时间", func=self.get_current_time, description="获取当前时间，无需输入参数"),
                Tool(name="计算器", func=self.calculator, description="执行数学计算，例如: 2+3*4"),
            ]

            # 添加天气工具（需要API Key）
            api_key = input("请输入心知天气API Key（没有则跳过，输入n）: ").strip()
            if api_key and api_key.lower() != 'n':
                tools.append(Tool(name="天气查询", func=self.create_weather_tool(api_key),
                                  description="查询城市天气，输入城市名称"))

            # ReAct提示模板
            template = """请尽可能好地回答以下问题。如果需要，可以适当的使用一些功能。

你有以下工具可用：
{tools}

请使用以下格式：
Question: 需要回答的问题。
Thought: 总是考虑应该做什么以及使用哪些工具。
Action: 应采取的行动，应为 [{tool_names}] 中的一个。
Action Input: 行动的输入。
Observation: 行动的结果。
...（这个过程可以重复多次）。
Thought: 我现在知道最终答案了。
Final Answer: 对原问题的最终答案。

开始！
Question: {input}
Thought: {agent_scratchpad}
"""

            prompt = PromptTemplate.from_template(template)
            agent = create_react_agent(llm, tools, prompt, stop_sequence=["\nobserv"])
            self.agent_executor = AgentExecutor.from_agent_and_tools(
                agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
            )
            print("Agent初始化完成")
            return True
        except Exception as e:
            print(f"Agent初始化失败: {e}")
            return False

    def get_current_time(self, _=None) -> str:
        """获取当前时间"""
        return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

    def calculator(self, expression: str) -> str:
        """计算器"""
        try:
            # 安全评估表达式
            allowed_names = {"abs": abs, "round": round, "max": max, "min": min}
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算失败: {e}"

    def create_weather_tool(self, api_key):
        """创建天气查询工具"""
        def weather_query(city: str):
            try:
                city = city.split("\n")[0]
                url = f"https://api.seniverse.com/v3/weather/now.json?key={api_key}&location={city}&language=zh-Hans&unit=c"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    weather = data["results"][0]["now"]["text"]
                    tem = data["results"][0]["now"]["temperature"]
                    return f"{city}的天气是{weather}，温度是{tem}℃"
                return f"无法获取{city}的天气信息"
            except Exception as e:
                return f"查询失败: {e}"
        return weather_query

    def query(self, question: str) -> str:
        """查询"""
        if self.agent_executor is None:
            return "Agent未初始化"

        try:
            result = self.agent_executor.invoke({"input": question})
            return result.get("output", "")
        except Exception as e:
            return f"查询失败: {e}"

    def run_demo(self):
        """运行演示"""
        if not self.init_agent():
            return

        print("\n=== Agent代理演示 ===")
        print("示例问题:")
        print("1. 现在几点了？")
        print("2. 计算 2 + 3 * 4")
        print("3. 成都天气怎么样（需要API Key）")
        print("输入 'quit' 退出")

        while True:
            question = input("\n问题: ")
            if question.lower() in ['quit', 'exit', 'q']:
                break

            print("思考中...")
            answer = self.query(question)
            print(f"回答: {answer}")

# ==================== FastAPI后端服务 ====================
class FastAPIServer:
    """FastAPI后端服务"""

    def __init__(self):
        self.app = None
        self.server_thread = None
        self.running = False

    def create_app(self):
        """创建FastAPI应用"""
        try:
            from fastapi import FastAPI, Body
            from fastapi.responses import StreamingResponse
            from openai import AsyncOpenAI
            from typing import List
            import uvicorn

            app = FastAPI(title="ChatBot API", description="基于Qwen2.5的聊天机器人")

            # 初始化OpenAI客户端
            client = AsyncOpenAI(api_key=Config.API_KEY, base_url=Config.OLLAMA_URL)

            @app.get("/")
            async def root():
                return {"message": "ChatBot API服务运行中", "status": "ok"}

            @app.post("/chat")
            async def chat(
                query: str = Body(..., description="用户输入"),
                sys_prompt: str = Body("你是一个有用的助手。", description="系统提示词"),
                history: List = Body([], description="历史对话"),
                history_len: int = Body(1, description="保留历史对话轮数"),
                temperature: float = Body(0.5, description="采样温度"),
                top_p: float = Body(0.5, description="采样概率"),
                max_tokens: int = Body(1024, description="最大token数"),
                stream: bool = Body(True, description="是否流式输出")
            ):
                # 控制历史记录长度
                if history_len > 0:
                    history = history[-2 * history_len:]

                # 构建消息
                messages = [{"role": "system", "content": sys_prompt}]
                messages.extend(history)
                messages.append({"role": "user", "content": query})

                # 发送请求
                response = await client.chat.completions.create(
                    model=Config.MODEL_NAME,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=stream
                )

                if stream:
                    async def generate():
                        async for chunk in response:
                            chunk_msg = chunk.choices[0].delta.content
                            if chunk_msg:
                                yield chunk_msg
                    return StreamingResponse(generate(), media_type="text/plain")
                else:
                    return {"response": response.choices[0].message.content}

            @app.get("/health")
            async def health():
                return {"status": "ok"}

            self.app = app
            return True
        except Exception as e:
            print(f"创建FastAPI应用失败: {e}")
            return False

    def start(self):
        """启动服务"""
        if not self.create_app():
            return False

        try:
            import uvicorn

            def run_server():
                uvicorn.run(self.app, host="0.0.0.0", port=Config.BACKEND_PORT, log_level="warning")

            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            self.running = True
            print(f"后端服务已启动: http://localhost:{Config.BACKEND_PORT}")
            return True
        except Exception as e:
            print(f"启动服务失败: {e}")
            return False

    def stop(self):
        """停止服务"""
        self.running = False
        print("服务已停止")

# ==================== Streamlit前端 ====================
class StreamlitUI:
    """Streamlit前端"""

    @staticmethod
    def get_code():
        """获取Streamlit前端代码"""
        return '''
import streamlit as st
import requests

BACKEND_URL = "http://localhost:6066/chat"

st.set_page_config(page_title="ChatBot", page_icon="🤖", layout="centered")
st.title("🤖 聊天机器人")

def clear_chat():
    st.session_state.history = []

with st.sidebar:
    st.title("⚙️ 设置")
    sys_prompt = st.text_input("系统提示词:", value="You are a helpful assistant.")
    history_len = st.slider("保留历史对话轮数:", 1, 10, 1)
    temperature = st.slider("temperature:", 0.01, 2.0, 0.5, 0.01)
    top_p = st.slider("top_p:", 0.01, 1.0, 0.5, 0.01)
    max_tokens = st.slider("max_tokens:", 256, 4096, 1024, 8)
    stream = st.checkbox("流式输出", True)
    st.button("清空聊天历史", on_click=clear_chat)

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("来和我聊天~~~"):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    data = {
        "query": prompt,
        "sys_prompt": sys_prompt,
        "history_len": history_len,
        "history": st.session_state.history,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": stream
    }
    
    response = requests.post(BACKEND_URL, json=data, stream=True)
    
    if response.status_code == 200:
        chunks = ""
        placeholder = st.chat_message("assistant")
        text = placeholder.markdown("")
        
        if stream:
            for chunk in response.iter_content(decode_unicode=True):
                chunks += chunk
                text.markdown(chunks)
        else:
            chunks = response.json().get("response", "")
            text.markdown(chunks)
        
        st.session_state.history.append({"role": "user", "content": prompt})
        st.session_state.history.append({"role": "assistant", "content": chunks})
'''

    @staticmethod
    def run():
        """运行Streamlit"""
        import tempfile
        code = StreamlitUI.get_code()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name

        print(f"启动Streamlit前端...")
        os.system(f'streamlit run "{temp_file}"')

# ==================== 性能测试模块 ====================
class PerformanceTest:
    """性能测试"""

    def test_transformers(self):
        """测试Transformers推理速度"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            print("\n=== Transformers推理测试 ===")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            print("加载模型...")
            tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)
            model = AutoModelForCausalLM.from_pretrained(
                Config.MODEL_PATH,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to(device)

            prompt = "讲个简短的故事"
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(device)

            # 预热
            _ = model.generate(model_inputs.input_ids, max_new_tokens=50)

            # 正式测试
            start = time.time()
            generated = model.generate(model_inputs.input_ids, max_new_tokens=100)
            end = time.time()

            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids
                           in zip(model_inputs.input_ids, generated)]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            num_tokens = len(generated_ids[0])
            elapsed = end - start
            print(f"生成token数: {num_tokens}")
            print(f"耗时: {elapsed:.2f}秒")
            print(f"速度: {num_tokens/elapsed:.2f} tokens/秒")
            print(f"响应: {response[:100]}...")
        except Exception as e:
            print(f"测试失败: {e}")

    def test_ollama(self):
        """测试Ollama API速度"""
        try:
            from openai import OpenAI

            print("\n=== Ollama API测试 ===")
            client = OpenAI(api_key=Config.API_KEY, base_url=Config.OLLAMA_URL)

            prompt = "讲个简短的故事"

            # 测试非流式
            start = time.time()
            response = client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                stream=False
            )
            end = time.time()

            content = response.choices[0].message.content
            num_tokens = len(content)
            elapsed = end - start
            print(f"非流式 - 生成token数: {num_tokens}, 耗时: {elapsed:.2f}秒, 速度: {num_tokens/elapsed:.2f} tokens/秒")

            # 测试流式
            start = time.time()
            response = client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                stream=True
            )
            content = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content
            end = time.time()

            num_tokens = len(content)
            elapsed = end - start
            print(f"流式 - 生成token数: {num_tokens}, 耗时: {elapsed:.2f}秒, 速度: {num_tokens/elapsed:.2f} tokens/秒")
        except Exception as e:
            print(f"测试失败: {e}")

# ==================== 主程序 ====================
def download_models():
    """下载模型"""
    try:
        from modelscope.hub.snapshot_download import snapshot_download

        print("正在下载Qwen2.5模型...")
        snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', cache_dir='models')

        print("正在下载Embedding模型...")
        snapshot_download('AI-ModelScope/bge-large-zh-v1.5', cache_dir='models')

        print("模型下载完成！")
    except Exception as e:
        print(f"下载失败: {e}")
        print("请手动下载或确保网络连接正常")

def start_web_service():
    """启动Web服务"""
    # 启动后端
    server = FastAPIServer()
    if not server.start():
        print("后端启动失败")
        return

    time.sleep(2)

    # 启动前端
    print("\n启动Streamlit前端...")
    StreamlitUI.run()

def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║    基于Qwen2.5的RAG+Agent完整项目                            ║
║    功能: 基础推理 | RAG问答 | Agent代理 | Web服务            ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # 检查依赖
    if not check_dependencies():
        return

    print("\n请选择功能:")
    print("1. 下载模型")
    print("2. 基础推理演示")
    print("3. RAG问答演示")
    print("4. Agent代理演示")
    print("5. 启动Web服务（FastAPI + Streamlit）")
    print("6. 性能测试")
    print("7. 退出")

    choice = input("\n请输入选项 (1-7): ").strip()

    if choice == "1":
        download_models()
    elif choice == "2":
        BasicInference().run_demo()
    elif choice == "3":
        RAGQA().run_demo()
    elif choice == "4":
        AgentDemo().run_demo()
    elif choice == "5":
        start_web_service()
    elif choice == "6":
        test = PerformanceTest()
        print("\n选择测试方式:")
        print("1. Transformers本地推理")
        print("2. Ollama API")
        sub = input("请选择 (1/2): ").strip()
        if sub == "1":
            test.test_transformers()
        else:
            test.test_ollama()
    else:
        print("再见！")

if __name__ == "__main__":
    main()