"""
完整RAG+Agent项目 - 基于Qwen2.5
运行方式：
1. 确保ollama已启动: ollama run qwen2.5:0.5b
2. 运行本文件: python app.py
3. 选择功能:
   - 选项1: 基础推理
   - 选项2: 启动Web服务（FastAPI + Streamlit）
   - 选项3: 成语接龙游戏
   - 选项4: Agent代理
   - 选项5: 性能测试
"""

import os
import sys
import json
import time
import re
import random
import requests
import threading
import webbrowser
from typing import List, Dict, Optional, Set
from datetime import datetime
from collections import defaultdict


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
        print(
            "pip install torch transformers langchain langchain-openai langchain-community langchain-huggingface fastapi uvicorn streamlit openai sentence-transformers faiss-cpu")
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

    # 成语接龙配置
    IDIOM_FILE = "cyjl.txt"  # 成语文档路径


# ==================== 基础推理模块 ====================
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


# ==================== 成语接龙游戏模块 ====================
class IdiomSolitaire:
    """成语接龙游戏 - 基于成语文档"""

    def __init__(self):
        self.idioms: Set[str] = set()  # 所有成语集合
        self.first_char_map: Dict[str, List[str]] = defaultdict(list)  # 按首字母分组
        self.load_idioms()

    def load_idioms(self) -> bool:
        """加载成语文档"""
        file_path = Config.IDIOM_FILE
        try:
            if not os.path.exists(file_path):
                print(f"⚠️ 文件不存在: {file_path}")
                print("请创建 cyjl.txt 文件，每行一个成语")
                print("示例格式:")
                print("一心一意")
                print("意气风发")
                print("发奋图强")
                return False

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # 提取成语（可能包含解释，取第一个词）
                    parts = line.split()
                    idiom = parts[0].strip()

                    # 验证成语格式（至少2个字，且都是中文）
                    if len(idiom) >= 2 and re.match(r'^[\u4e00-\u9fff]+$', idiom):
                        self.idioms.add(idiom)
                        first_char = idiom[0]
                        self.first_char_map[first_char].append(idiom)

            print(f"✅ 加载成语完成，共 {len(self.idioms)} 个成语")
            return True

        except Exception as e:
            print(f"❌ 加载成语失败: {e}")
            return False

    def is_valid_idiom(self, idiom: str) -> bool:
        """检查成语是否在文档中"""
        return idiom in self.idioms

    def get_idioms_by_first_char(self, char: str) -> List[str]:
        """根据首字获取成语列表"""
        return self.first_char_map.get(char, [])

    def get_last_char(self, idiom: str) -> str:
        """获取成语的最后一个字"""
        return idiom[-1] if idiom else ""

    def check_connection(self, prev_idiom: str, next_idiom: str) -> bool:
        """检查两个成语是否能够接龙"""
        if not prev_idiom or not next_idiom:
            return False
        return prev_idiom[-1] == next_idiom[0]

    def get_available_idioms(self, start_char: str, used_idioms: Set[str]) -> List[str]:
        """获取可用的成语列表"""
        available = self.get_idioms_by_first_char(start_char)
        return [a for a in available if a not in used_idioms]

    def ai_choose_idiom(self, start_char: str, used_idioms: Set[str], client) -> Optional[str]:
        """AI选择接龙成语"""
        available = self.get_available_idioms(start_char, used_idioms)

        if not available:
            return None

        # 如果可用成语较少，直接随机选择
        if len(available) <= 5:
            return random.choice(available)

        # 让AI选择合适的成语
        prompt = f"""你正在玩成语接龙游戏。需要接龙以"{start_char}"开头的成语。
请从以下成语中选择一个进行接龙，只回复一个成语，不要有其他内容：

可选的成语有: {available[:30]}

请选择一个成语回复："""

        try:
            response = client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个成语接龙专家。请只回复一个成语，不要有其他内容。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.5
            )

            ai_idiom = response.choices[0].message.content.strip()
            # 清理可能的标点符号
            ai_idiom = re.sub(r'[^\u4e00-\u9fff]', '', ai_idiom)

            # 验证AI回答的成语
            if ai_idiom in available:
                return ai_idiom
            else:
                # 如果AI选的成语无效，随机选一个
                return random.choice(available) if available else None

        except Exception as e:
            print(f"AI调用失败: {e}")
            return random.choice(available) if available else None

    def run_game(self):
        """运行成语接龙游戏（命令行版本）"""
        if not self.idioms:
            print("❌ 成语文档未加载，请确保 cyjl.txt 文件存在")
            return

        print("\n" + "=" * 50)
        print("🎮 成语接龙游戏")
        print("=" * 50)
        print("游戏规则:")
        print("1. 玩家输入一个成语，AI接龙")
        print("2. AI回答的成语必须以玩家成语的最后一个字开头")
        print("3. 玩家继续接AI的成语，如此轮流进行")
        print("4. 所有成语必须在成语文档中存在")
        print("5. 如果某方无法接龙，则判负")
        print("6. 输入 'quit' 退出游戏")
        print("=" * 50)

        # 初始化LLM客户端
        try:
            from openai import OpenAI
            client = OpenAI(api_key=Config.API_KEY, base_url=Config.OLLAMA_URL)
            print("✅ AI已准备就绪")
        except Exception as e:
            print(f"❌ AI连接失败: {e}")
            return

        # 游戏主循环 - 每局结束后可以重新开始
        while True:
            used_idioms = set()  # 记录已使用的成语
            current_last_char = None
            turn = "player"  # player 或 ai

            # 随机决定谁先开始
            if random.choice([True, False]):
                print("\n🎲 玩家先开始！")
                turn = "player"
            else:
                print("\n🎲 AI先开始！")
                turn = "ai"

            # 开始一局游戏
            game_over = False
            while not game_over:
                if turn == "player":
                    print("\n" + "-" * 30)
                    player_input = input("🎯 请输入成语: ").strip()

                    if player_input.lower() in ['quit', 'exit', 'q']:
                        print("👋 游戏结束，再见！")
                        return

                    # 验证玩家输入的成语
                    if not self.is_valid_idiom(player_input):
                        print(f"❌ 错误: '{player_input}' 不在成语文档中，你输了！")
                        game_over = True
                        break

                    # 检查是否重复使用
                    if player_input in used_idioms:
                        print(f"⚠️  '{player_input}' 已经使用过了，你输了！")
                        game_over = True
                        break

                    # 检查接龙规则（如果不是第一轮）
                    if current_last_char is not None:
                        if not self.check_connection(current_last_char, player_input):
                            print(f"❌ 错误: '{player_input}' 不以 '{current_last_char[-1]}' 开头，你输了！")
                            game_over = True
                            break

                    print(f"✅ 玩家: {player_input}")
                    used_idioms.add(player_input)
                    current_last_char = player_input
                    turn = "ai"

                else:  # AI回合
                    if current_last_char is None:
                        # AI先手，随机选一个成语
                        print("\n🤖 AI思考中...")
                        available = list(self.idioms)
                        # 排除已使用的
                        available = [a for a in available if a not in used_idioms]
                        ai_idiom = random.choice(available) if available else None
                    else:
                        # AI需要接龙
                        print("\n🤖 AI思考中...")
                        last_char = self.get_last_char(current_last_char)
                        ai_idiom = self.ai_choose_idiom(last_char, used_idioms, client)

                    if ai_idiom is None:
                        print(f"❌ AI无法接龙，AI认输！玩家获胜！")
                        game_over = True
                        break

                    # 验证AI回答
                    if not self.is_valid_idiom(ai_idiom):
                        print(f"❌ AI回答 '{ai_idiom}' 不在成语文档中，AI判负！玩家获胜！")
                        game_over = True
                        break

                    if ai_idiom in used_idioms:
                        print(f"⚠️  AI回答 '{ai_idiom}' 已经使用过了，AI判负！玩家获胜！")
                        game_over = True
                        break

                    # 检查接龙规则（如果不是第一轮）
                    if current_last_char is not None:
                        last_char = self.get_last_char(current_last_char)
                        if not self.check_connection(last_char, ai_idiom):
                            print(f"❌ AI回答 '{ai_idiom}' 不以 '{last_char}' 开头，AI判负！玩家获胜！")
                            game_over = True
                            break

                    print(f"🤖 AI: {ai_idiom}")
                    used_idioms.add(ai_idiom)
                    current_last_char = ai_idiom
                    turn = "player"

                    # 显示剩余可接龙的成语数量
                    if current_last_char:
                        remaining = self.get_available_idioms(self.get_last_char(current_last_char), used_idioms)
                        print(f"📊 剩余可接龙成语: {len(remaining)} 个")

            # 游戏结束，询问是否重新开始
            print("\n" + "=" * 30)
            replay = input("🎮 是否开始新的一局？(y/n): ").strip().lower()
            if replay != 'y':
                print("👋 感谢游玩，再见！")
                break
            print("\n" + "=" * 50)
            print("🎮 新的一局开始！")
            print("=" * 50)

    def run_demo(self):
        """运行游戏演示（与接口兼容）"""
        self.run_game()

    # ==================== Web服务接口方法 ====================
    def get_game_state(self, used_idioms: list, last_idiom: str = None, current_turn: str = "player") -> dict:
        """获取游戏状态"""
        return {
            "idioms_count": len(self.idioms),
            "used_idioms": used_idioms,
            "last_idiom": last_idiom,
            "current_turn": current_turn,
            "last_char": self.get_last_char(last_idiom) if last_idiom else None
        }

    def validate_player_idiom(self, idiom: str, last_idiom: str = None, used_idioms: Set[str] = None) -> dict:
        """验证玩家输入的成语"""
        if not idiom:
            return {"valid": False, "reason": "请输入成语"}

        if not self.is_valid_idiom(idiom):
            return {"valid": False, "reason": f"'{idiom}' 不在成语文档中"}

        if used_idioms and idiom in used_idioms:
            return {"valid": False, "reason": f"'{idiom}' 已经使用过了"}

        if last_idiom:
            expected_char = self.get_last_char(last_idiom)
            if idiom[0] != expected_char:
                return {"valid": False, "reason": f"'{idiom}' 不以 '{expected_char}' 开头"}

        return {"valid": True, "reason": ""}

    def get_ai_idiom(self, last_char: str, used_idioms: Set[str]) -> dict:
        """获取AI的接龙成语（供Web服务使用）"""
        available = self.get_available_idioms(last_char, used_idioms)

        if not available:
            return {"success": False, "idiom": None, "reason": f"没有以 '{last_char}' 开头的可用成语"}

        # 初始化LLM客户端
        try:
            from openai import OpenAI
            client = OpenAI(api_key=Config.API_KEY, base_url=Config.OLLAMA_URL)
        except Exception as e:
            # 如果无法连接AI，随机选择一个
            return {"success": True, "idiom": random.choice(available), "reason": ""}

        # 如果可用成语较少，直接随机选择
        if len(available) <= 5:
            return {"success": True, "idiom": random.choice(available), "reason": ""}

        # 让AI选择合适的成语
        prompt = f"""你正在玩成语接龙游戏。需要接龙以"{last_char}"开头的成语。
请从以下成语中选择一个进行接龙，只回复一个成语，不要有其他内容：

可选的成语有: {available[:30]}

请选择一个成语回复："""

        try:
            response = client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个成语接龙专家。请只回复一个成语，不要有其他内容。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.5
            )

            ai_idiom = response.choices[0].message.content.strip()
            # 清理可能的标点符号
            ai_idiom = re.sub(r'[^\u4e00-\u9fff]', '', ai_idiom)

            # 验证AI回答的成语
            if ai_idiom in available:
                return {"success": True, "idiom": ai_idiom, "reason": ""}
            else:
                # 如果AI选的成语无效，随机选一个
                return {"success": True, "idiom": random.choice(available), "reason": ""}

        except Exception as e:
            return {"success": True, "idiom": random.choice(available), "reason": ""}


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
        self.idiom_game = IdiomSolitaire()  # 创建成语接龙游戏实例

    def create_app(self):
        """创建FastAPI应用"""
        try:
            from fastapi import FastAPI, Body
            from fastapi.responses import StreamingResponse
            from fastapi.middleware.cors import CORSMiddleware
            from openai import AsyncOpenAI
            from typing import List
            import uvicorn

            app = FastAPI(title="ChatBot API", description="基于Qwen2.5的聊天机器人")

            # 添加CORS中间件
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

            # 初始化OpenAI客户端
            client = AsyncOpenAI(api_key=Config.API_KEY, base_url=Config.OLLAMA_URL)

            @app.get("/")
            async def root():
                return {"message": "ChatBot API服务运行中", "status": "ok"}

            # ==================== 聊天API ====================
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

            # ==================== 成语接龙API ====================
            @app.get("/idiom/game_state")
            async def get_game_state():
                """获取游戏初始状态"""
                return {
                    "success": True,
                    "idioms_count": len(self.idiom_game.idioms),
                    "message": f"成语库加载完成，共{len(self.idiom_game.idioms)}个成语"
                }

            @app.post("/idiom/validate")
            async def validate_idiom(
                    idiom: str = Body(..., description="玩家输入的成语"),
                    last_idiom: str = Body(None, description="上一个成语"),
                    used_idioms: List[str] = Body([], description="已使用的成语列表")
            ):
                """验证玩家输入的成语"""
                result = self.idiom_game.validate_player_idiom(idiom, last_idiom, set(used_idioms))
                return result

            @app.post("/idiom/ai_move")
            async def ai_move(
                    last_char: str = Body(..., description="需要接龙的首字"),
                    used_idioms: List[str] = Body([], description="已使用的成语列表")
            ):
                """获取AI的接龙成语"""
                result = self.idiom_game.get_ai_idiom(last_char, set(used_idioms))
                return result

            @app.get("/idiom/available")
            async def get_available(
                    start_char: str,
                    used_idioms: List[str] = Body([], description="已使用的成语列表")
            ):
                """获取可用的成语列表"""
                available = self.idiom_game.get_available_idioms(start_char, set(used_idioms))
                return {
                    "success": True,
                    "available_count": len(available),
                    "available": available[:50]  # 只返回前50个
                }

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
        """获取Streamlit前端代码（包含成语接龙）"""
        return '''
import streamlit as st
import requests
import json
import random

BACKEND_URL = "http://localhost:6066"

st.set_page_config(page_title="智能应用平台", page_icon="🤖", layout="wide")

# 侧边栏选择功能
st.sidebar.title("📋 功能选择")
app_mode = st.sidebar.radio(
    "选择功能",
    ["💬 智能聊天", "🎮 成语接龙游戏"],
    index=0
)

# ==================== 智能聊天模块 ====================
if app_mode == "💬 智能聊天":
    st.title("🤖 智能聊天机器人")
    st.caption("基于Qwen2.5的对话助手")

    def clear_chat():
        st.session_state.chat_history = []

    with st.sidebar:
        st.markdown("---")
        st.title("⚙️ 聊天设置")
        sys_prompt = st.text_input("系统提示词:", value="You are a helpful assistant.")
        history_len = st.slider("保留历史对话轮数:", 1, 10, 1)
        temperature = st.slider("temperature:", 0.01, 2.0, 0.5, 0.01)
        top_p = st.slider("top_p:", 0.01, 1.0, 0.5, 0.01)
        max_tokens = st.slider("max_tokens:", 256, 4096, 1024, 8)
        stream = st.checkbox("流式输出", True)
        st.button("清空聊天历史", on_click=clear_chat)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 显示聊天历史
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 聊天输入
    if prompt := st.chat_input("来和我聊天~~~"):
        with st.chat_message("user"):
            st.markdown(prompt)

        data = {
            "query": prompt,
            "sys_prompt": sys_prompt,
            "history_len": history_len,
            "history": st.session_state.chat_history,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": stream
        }

        response = requests.post(f"{BACKEND_URL}/chat", json=data, stream=True)

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

            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.chat_history.append({"role": "assistant", "content": chunks})

# ==================== 成语接龙游戏模块 ====================
else:
    st.title("🎮 成语接龙游戏")
    st.caption("与AI进行成语接龙对决，轮流接龙，无法接龙者判负！")

    # 初始化游戏状态
    def reset_game():
        """重置游戏状态"""
        st.session_state.used_idioms = []
        st.session_state.last_idiom = None
        st.session_state.game_over = False
        st.session_state.winner = None
        st.session_state.game_message = ""
        # 随机决定谁先开始
        if st.session_state.get("random_turn", True):
            st.session_state.current_turn = random.choice(["player", "ai"])
            st.session_state.game_message = f"🎲 {'玩家' if st.session_state.current_turn == 'player' else 'AI'}先开始！"
        else:
            st.session_state.current_turn = "player"
            st.session_state.game_message = "🎲 玩家先开始！"
        st.session_state.waiting_for_ai = False
        st.session_state.player_input_value = ""  # 清空输入框的值

    # 初始化session state
    if "game_initialized" not in st.session_state:
        st.session_state.game_initialized = False
        st.session_state.used_idioms = []
        st.session_state.last_idiom = None
        st.session_state.current_turn = "player"
        st.session_state.game_over = False
        st.session_state.winner = None
        st.session_state.game_message = ""
        st.session_state.idioms_count = 0
        st.session_state.random_turn = True
        st.session_state.waiting_for_ai = False
        st.session_state.player_input_value = ""

    # 初始化游戏（加载成语库）
    if not st.session_state.game_initialized:
        with st.spinner("加载成语库..."):
            try:
                response = requests.get(f"{BACKEND_URL}/idiom/game_state")
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.idioms_count = data.get("idioms_count", 0)
                    st.session_state.game_initialized = True
                    # 随机决定先手
                    if st.session_state.random_turn:
                        st.session_state.current_turn = random.choice(["player", "ai"])
                        st.session_state.game_message = f"🎲 {'玩家' if st.session_state.current_turn == 'player' else 'AI'}先开始！"
                    else:
                        st.session_state.current_turn = "player"
                        st.session_state.game_message = "🎲 玩家先开始！"
                    st.success(f"✅ 成语库加载完成，共{st.session_state.idioms_count}个成语！")
                else:
                    st.error("无法连接到后端服务，请确保后端已启动")
            except Exception as e:
                st.error(f"连接失败: {e}")

    # 游戏规则说明
    with st.expander("📖 游戏规则"):
        st.markdown("""
        **成语接龙游戏规则：**
        1. 玩家和AI轮流进行成语接龙
        2. 每个成语的最后一个字必须与下一个成语的第一个字相同
        3. 所有成语必须在成语库中存在
        4. 已经使用过的成语不能重复使用
        5. 如果某方无法接龙（没有可用的成语），则判负
        6. 游戏结束后会自动重置并开始新的一局
        """)

    # 侧边栏游戏信息
    with st.sidebar:
        st.markdown("---")
        st.subheader("🎮 游戏信息")
        st.metric("成语库数量", st.session_state.idioms_count)
        st.metric("已使用成语", len(st.session_state.used_idioms))
        
        if st.session_state.last_idiom and not st.session_state.game_over:
            last_char = st.session_state.last_idiom[-1]
            st.info(f"📝 最后一个字: **{last_char}**")
            st.info(f"🎯 需要以 **{last_char}** 开头接龙")
        
        st.markdown("---")
        
        # 游戏设置
        st.subheader("⚙️ 游戏设置")
        st.session_state.random_turn = st.checkbox("随机决定先手", value=st.session_state.random_turn)
        
        # 重置游戏按钮
        if st.button("🔄 新的一局"):
            reset_game()
            st.rerun()
        
        # 显示已使用的成语
        if st.session_state.used_idioms:
            st.markdown("---")
            st.subheader("📜 已使用成语")
            for i, idiom in enumerate(st.session_state.used_idioms[-15:], 1):
                st.text(f"{i}. {idiom}")

    # 游戏主界面 - 使用两列布局
    col1, col2 = st.columns([3, 1])

    with col1:
        # 显示游戏状态区域
        game_status_placeholder = st.empty()
        
        # 显示游戏结果和当前状态
        if st.session_state.game_over:
            if st.session_state.winner == "player":
                game_status_placeholder.success(f"🎉 恭喜！玩家获胜！{st.session_state.game_message}")
            elif st.session_state.winner == "ai":
                game_status_placeholder.error(f"😢 AI获胜！{st.session_state.game_message}")
            else:
                game_status_placeholder.info(f"游戏结束: {st.session_state.game_message}")
            
            # 游戏结束后显示开始新游戏按钮
            if st.button("🎮 开始新游戏", key="new_game_btn", use_container_width=True):
                reset_game()
                st.rerun()
        else:
            # 显示当前回合
            if st.session_state.current_turn == "player":
                game_status_placeholder.info("🎯 **当前回合：玩家**")
            else:
                game_status_placeholder.warning("🤖 **当前回合：AI思考中...**")
            
            # 显示上一个成语
            if st.session_state.last_idiom:
                st.success(f"📖 上一个成语: **{st.session_state.last_idiom}**")
                st.info(f"🔤 需要以 **{st.session_state.last_idiom[-1]}** 开头的成语")
            else:
                st.info("🎲 游戏开始，请输入第一个成语")

    with col2:
        # 玩家输入区域
        if not st.session_state.game_over:
            if st.session_state.current_turn == "player" and not st.session_state.waiting_for_ai:
                # 使用 session_state 来存储输入框的值，避免被清空
                player_input = st.text_input(
                    "📝 请输入成语:", 
                    key="player_input_widget",
                    value=st.session_state.player_input_value,
                    placeholder="例如: 一心一意"
                )
                submit = st.button("✅ 提交", type="primary", use_container_width=True)
                
                if submit and player_input:
                    with st.spinner("验证中..."):
                        # 验证玩家输入的成语
                        validate_data = {
                            "idiom": player_input,
                            "last_idiom": st.session_state.last_idiom,
                            "used_idioms": st.session_state.used_idioms
                        }
                        response = requests.post(f"{BACKEND_URL}/idiom/validate", json=validate_data)
                        
                        if response.status_code == 200:
                            result = response.json()
                            if result["valid"]:
                                # 有效的成语
                                st.session_state.used_idioms.append(player_input)
                                st.session_state.last_idiom = player_input
                                st.session_state.game_message = f"玩家输入: {player_input}"
                                st.session_state.player_input_value = ""  # 清空输入
                                
                                # 检查AI是否还能接龙
                                last_char = player_input[-1]
                                ai_check_data = {
                                    "last_char": last_char,
                                    "used_idioms": st.session_state.used_idioms
                                }
                                ai_response = requests.post(f"{BACKEND_URL}/idiom/ai_move", json=ai_check_data)
                                
                                if ai_response.status_code == 200:
                                    ai_result = ai_response.json()
                                    if not ai_result["success"] or ai_result["idiom"] is None:
                                        # AI无法接龙，玩家获胜，游戏结束并自动重置
                                        st.session_state.game_over = True
                                        st.session_state.winner = "player"
                                        st.session_state.game_message = "AI无法接龙，玩家获胜！"
                                        st.success("🎉 恭喜获胜！即将开始新游戏...")
                                        st.rerun()
                                    else:
                                        # 切换到AI回合
                                        st.session_state.current_turn = "ai"
                                        st.session_state.waiting_for_ai = True
                                        st.rerun()
                            else:
                                st.error(f"❌ {result['reason']}")
                        else:
                            st.error("验证失败，请重试")
                
                # 提示信息
                if st.session_state.last_idiom:
                    st.caption(f"💡 提示：成语必须以 '{st.session_state.last_idiom[-1]}' 开头")
            else:
                # AI回合或等待AI响应
                if st.session_state.waiting_for_ai or st.session_state.current_turn == "ai":
                    with st.spinner("🤖 AI正在思考中..."):
                        # 执行AI回合
                        if st.session_state.last_idiom:
                            last_char = st.session_state.last_idiom[-1]
                            ai_data = {
                                "last_char": last_char,
                                "used_idioms": st.session_state.used_idioms
                            }
                        else:
                            # AI先手，随机选一个成语
                            ai_data = {
                                "last_char": "",
                                "used_idioms": st.session_state.used_idioms
                            }
                        
                        try:
                            response = requests.post(f"{BACKEND_URL}/idiom/ai_move", json=ai_data)
                            if response.status_code == 200:
                                result = response.json()
                                if result["success"] and result["idiom"]:
                                    ai_idiom = result["idiom"]
                                    st.session_state.used_idioms.append(ai_idiom)
                                    st.session_state.last_idiom = ai_idiom
                                    st.session_state.game_message = f"AI输入: {ai_idiom}"
                                    
                                    # 检查玩家是否还能接龙
                                    last_char = ai_idiom[-1]
                                    player_check_data = {
                                        "idiom": "",
                                        "last_idiom": ai_idiom,
                                        "used_idioms": st.session_state.used_idioms
                                    }
                                    # 获取可用的接龙成语
                                    available_response = requests.get(
                                        f"{BACKEND_URL}/idiom/available",
                                        params={"start_char": last_char},
                                        json={"used_idioms": st.session_state.used_idioms}
                                    )
                                    
                                    if available_response.status_code == 200:
                                        available_data = available_response.json()
                                        if available_data.get("available_count", 0) == 0:
                                            # 玩家无法接龙，AI获胜，游戏结束并自动重置
                                            st.session_state.game_over = True
                                            st.session_state.winner = "ai"
                                            st.session_state.game_message = "玩家无法接龙，AI获胜！"
                                            st.session_state.waiting_for_ai = False
                                            st.error("😢 AI获胜！即将开始新游戏...")
                                            st.rerun()
                                        else:
                                            # 切换到玩家回合
                                            st.session_state.current_turn = "player"
                                            st.session_state.waiting_for_ai = False
                                            st.rerun()
                                    else:
                                        st.session_state.current_turn = "player"
                                        st.session_state.waiting_for_ai = False
                                        st.rerun()
                                else:
                                    # AI无法接龙，玩家获胜
                                    st.session_state.game_over = True
                                    st.session_state.winner = "player"
                                    st.session_state.game_message = result.get("reason", "AI无法接龙，玩家获胜！")
                                    st.session_state.waiting_for_ai = False
                                    st.success("🎉 恭喜获胜！即将开始新游戏...")
                                    st.rerun()
                            else:
                                st.error("AI响应失败，请手动刷新")
                                st.session_state.current_turn = "player"
                                st.session_state.waiting_for_ai = False
                        except Exception as e:
                            st.error(f"AI响应异常: {e}")
                            st.session_state.current_turn = "player"
                            st.session_state.waiting_for_ai = False
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
            print(f"速度: {num_tokens / elapsed:.2f} tokens/秒")
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
            print(
                f"非流式 - 生成token数: {num_tokens}, 耗时: {elapsed:.2f}秒, 速度: {num_tokens / elapsed:.2f} tokens/秒")

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
            print(
                f"流式 - 生成token数: {num_tokens}, 耗时: {elapsed:.2f}秒, 速度: {num_tokens / elapsed:.2f} tokens/秒")
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
║    功能: 基础推理 | 成语接龙 | Agent代理 | Web服务           ║
╚══════════════════════════════════════════════════════════════╝
    """)

    print("\n请选择功能:")
    print("1. 下载模型")
    print("2. 基础推理演示")
    print("3. 成语接龙游戏")
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
        game = IdiomSolitaire()
        game.run_demo()
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