"""
完整RAG+Agent项目 - 集成5个新API（驾考、文本检测、经典语录）
支持：智能聊天、成语接龙、天气、Agent工具调用、语录独立页面
"""

import os
import re
import time
import random
import threading
import requests
import subprocess
import asyncio
from typing import List, Dict, Set, Any
from datetime import datetime
from collections import defaultdict
from pydantic import Field

# LangChain
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler

# ==================== 配置 ====================
class Config:
    DEEPSEEK_API_KEY = "sk-8454f30b99cc4305a20ea976f5c3f9ac"
    DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL = "deepseek-chat"
    BACKEND_PORT = 6066
    IDIOM_FILE = "cyjl.txt"
    WEATHER_API_KEY = "SXd4viN_pwVdcogzD"


# ==================== 原有工具函数 ====================
def query_weather(city: str) -> str:
    city = city.strip().split('\n')[0]
    url = f"https://api.seniverse.com/v3/weather/now.json?key={Config.WEATHER_API_KEY}&location={city}&language=zh-Hans&unit=c"
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            data = r.json()
            weather = data["results"][0]["now"]["text"]
            temp = data["results"][0]["now"]["temperature"]
            return f"{city}的天气是{weather}，温度{temp}℃"
        else:
            return f"无法获取{city}的天气"
    except Exception as e:
        return f"查询失败: {e}"

def get_current_time(_=None) -> str:
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

def calculate(expr: str) -> str:
    try:
        allowed = {"abs":abs, "round":round, "max":max, "min":min, "log": lambda x: __import__('math').log(x)}
        result = eval(expr, {"__builtins__":{}}, allowed)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {e}"


# ==================== 新增API工具函数 ====================
# ==================== 修复后的API工具函数（接受可选参数） ====================
# ==================== 修复后的API工具函数（接受可选参数） ====================
def driving_test_quiz(question: str) -> str:
    """驾考题库咨询 - 使用 POST 请求"""
    url = "https://api.pearktrue.cn/api/translate/ai/"

    # 尝试 POST JSON 格式
    try:
        r = requests.post(url, json={"text": question}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data.get("code") == 200:
                return data.get("data", "未获取到答案")
            else:
                # 尝试 GET 方式
                pass
        else:
            pass
    except Exception as e:
        pass

    # 备用：GET 方式，尝试不同参数名
    for param_name in ["text", "q", "query", "content", "question", "msg"]:
        try:
            r = requests.get(url, params={param_name: question}, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data.get("code") == 200:
                    return data.get("data", "未获取到答案")
                elif data.get("code") == 203:
                    # 参数名不对，继续尝试
                    continue
                else:
                    return f"接口返回: {data.get('msg', '未知错误')}"
        except:
            continue

    return "驾考咨询接口暂时无法使用，请稍后重试"

def text_security_check(text: str) -> str:
    """AI文本违规检测"""
    url = "https://api.pearktrue.cn/api/text_security"
    params = {"text": text}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data.get("code") == 200:
                result = data.get("data", {})
                return f"检测结果: {result.get('result', '未知')}，详情: {result.get('detail', '')}"
            else:
                return f"检测失败: {data.get('msg', '未知错误')}"
        else:
            return f"请求失败，状态码: {r.status_code}"
    except Exception as e:
        return f"文本检测接口异常: {e}"

def get_tiangou(_=None) -> str:
    """经典语录：舔狗（接受一个可选参数，忽略它）"""
    url = "https://api.pearktrue.cn/api/jdyl/tiangou.php"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.text.strip()
        else:
            return f"获取失败，状态码: {r.status_code}"
    except Exception as e:
        return f"舔狗语录接口异常: {e}"

def get_qinghua(_=None) -> str:
    """经典语录：情话（接受一个可选参数，忽略它）"""
    url = "https://api.pearktrue.cn/api/jdyl/qinghua.php"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.text.strip()
        else:
            return f"获取失败，状态码: {r.status_code}"
    except Exception as e:
        return f"情话语录接口异常: {e}"

def get_saohua(_=None) -> str:
    """经典语录：骚话（接受一个可选参数，忽略它）"""
    # 注意：骚话API地址为推测，如不正确可自行修改
    url = "https://api.pearktrue.cn/api/jdyl/saohua.php"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.text.strip()
        else:
            return f"获取失败，状态码: {r.status_code}"
    except Exception as e:
        return f"骚话语录接口异常: {e}"


# ==================== Agent回调处理器 ====================
class AgentStepCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.steps = []
    def on_agent_action(self, action, **kwargs):
        self.steps.append(f"**调用工具**: {action.tool}\n**输入**: {action.tool_input}\n**思考**: {action.log}")
    def on_tool_end(self, output, **kwargs):
        self.steps.append(f"**工具返回**: {output}")
    def on_agent_finish(self, finish, **kwargs):
        self.steps.append(f"**最终答案**: {finish.return_values['output']}")

class AgentDemo:
    def __init__(self):
        self.agent_executor = None
    def init_agent(self):
        try:
            tools = [
                Tool(name="weather_query", func=query_weather, description="查询城市天气"),
                Tool(name="current_time", func=get_current_time, description="获取当前时间"),
                Tool(name="calculator", func=calculate, description="数学计算"),
                Tool(name="driving_test", func=driving_test_quiz, description="驾考题库咨询，输入驾考相关的问题"),
                Tool(name="text_security", func=text_security_check, description="检测文本是否包含违规内容，输入需要检测的文本"),
                Tool(name="tiangou", func=get_tiangou, description="随机获取一条舔狗语录，无需输入参数"),
                Tool(name="qinghua", func=get_qinghua, description="随机获取一条情话语录，无需输入参数"),
                Tool(name="saohua", func=get_saohua, description="随机获取一条骚话语录，无需输入参数"),
            ]
            llm = ChatOpenAI(
                openai_api_key=Config.DEEPSEEK_API_KEY,
                base_url=Config.DEEPSEEK_BASE_URL,
                model=Config.DEEPSEEK_MODEL,
                temperature=0.5,
                top_p=0.9
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个有用的AI助手，可以使用以下工具：weather_query（天气）、current_time（时间）、calculator（计算）、driving_test（驾考咨询）、text_security（文本违规检测）、tiangou（舔狗语录）、qinghua（情话语录）、saohua（骚话语录）。请根据用户的问题，选择合适的工具。"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            agent = create_tool_calling_agent(llm, tools, prompt)
            self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=3)
            print("✅ Agent 初始化完成（已集成5个新API工具）")
            return True
        except Exception as e:
            print(f"Agent初始化失败: {e}")
            return False
    def query_with_steps(self, question: str):
        if not self.agent_executor:
            return {"answer": "Agent未初始化", "steps": []}
        cb = AgentStepCallbackHandler()
        try:
            result = self.agent_executor.invoke({"input": question}, config={"callbacks": [cb]})
            return {"answer": result.get("output", ""), "steps": cb.steps}
        except Exception as e:
            return {"answer": f"错误: {e}", "steps": []}
    def run_demo(self):
        if not self.init_agent():
            return
        print("\nAgent命令行模式，输入quit退出")
        while True:
            q = input("问题: ")
            if q.lower() in ('quit','exit'): break
            res = self.query_with_steps(q)
            print(f"答案: {res['answer']}")


# ==================== 成语接龙（简化，同前）====================
class IdiomSolitaire:
    def __init__(self):
        self.idioms = set()
        self.first_char_map = defaultdict(list)
        self.load_idioms()
    def load_idioms(self):
        if not os.path.exists(Config.IDIOM_FILE):
            default = ["一心一意","意气风发","发奋图强","强人所难","难能可贵","贵耳贱目","目中无人","人山人海"]
            for i in default:
                self.idioms.add(i)
                self.first_char_map[i[0]].append(i)
            return True
        with open(Config.IDIOM_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line=line.strip()
                if not line: continue
                w=line.split()[0]
                if len(w)>=2 and re.match(r'^[\u4e00-\u9fff]+$', w):
                    self.idioms.add(w)
                    self.first_char_map[w[0]].append(w)
        print(f"加载成语 {len(self.idioms)} 个")
        return True
    def is_valid(self, w): return w in self.idioms
    def last_char(self, w): return w[-1] if w else ""
    def validate(self, w, last, used):
        if not w: return (False, "空")
        if not self.is_valid(w): return (False, "不在库")
        if w in used: return (False, "已使用")
        if last and w[0] != self.last_char(last): return (False, f"不以'{self.last_char(last)}'开头")
        return (True, "")
    def ai_move(self, last_char, used):
        from openai import OpenAI
        client = OpenAI(api_key=Config.DEEPSEEK_API_KEY, base_url=Config.DEEPSEEK_BASE_URL)
        avail = [i for i in self.first_char_map.get(last_char,[]) if i not in used]
        if not avail: return None
        if len(avail)<=5: return random.choice(avail)
        prompt = f"需要接龙以'{last_char}'开头的成语。可选: {avail[:30]}。只回复一个成语："
        try:
            resp = client.chat.completions.create(model=Config.DEEPSEEK_MODEL, messages=[{"role":"system","content":"只回复成语"},{"role":"user","content":prompt}], max_tokens=20)
            ans = re.sub(r'[^\u4e00-\u9fff]','',resp.choices[0].message.content.strip())
            return ans if ans in avail else random.choice(avail)
        except:
            return random.choice(avail)
    def run_demo(self):
        from openai import OpenAI
        client = OpenAI(api_key=Config.DEEPSEEK_API_KEY, base_url=Config.DEEPSEEK_BASE_URL)
        print("\n成语接龙，输入quit退出")
        while True:
            used=set(); last=None; turn=random.choice(["player","ai"])
            print(f"\n{'玩家' if turn=='player' else 'AI'}先手")
            over=False
            while not over:
                if turn=="player":
                    w=input("成语: ").strip()
                    if w.lower() in ('quit','exit'): return
                    ok,reason=self.validate(w,last,used)
                    if not ok: print(f"❌ {reason}，你输了"); break
                    print(f"✅ {w}")
                    used.add(w); last=w; turn="ai"
                else:
                    print("AI思考...")
                    if last is None:
                        cand=[i for i in self.idioms if i not in used]
                        ai_w=random.choice(cand) if cand else None
                    else:
                        ai_w=self.ai_move(self.last_char(last), used)
                    if not ai_w: print("AI无法接龙，你赢了！"); break
                    if not self.is_valid(ai_w) or ai_w in used or (last and ai_w[0]!=self.last_char(last)):
                        print("AI违规，你赢了！"); break
                    print(f"🤖 {ai_w}")
                    used.add(ai_w); last=ai_w; turn="player"
            if input("新一局？(y/n): ").lower()!='y': break


# ==================== 基础推理（命令行）====================
class BasicInference:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=Config.DEEPSEEK_API_KEY, base_url=Config.DEEPSEEK_BASE_URL)
    def chat(self, prompt):
        resp = self.client.chat.completions.create(model=Config.DEEPSEEK_MODEL, messages=[{"role":"user","content":prompt}])
        return resp.choices[0].message.content
    def run_demo(self):
        print("\n基础推理（quit退出）")
        while True:
            q=input("你: ")
            if q.lower() in ('quit','exit'): break
            print("助手:", self.chat(q))


# ==================== FastAPI 后端 ====================
from fastapi import FastAPI, Body, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
import uvicorn

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

idiom_game = IdiomSolitaire()
agent = AgentDemo()
agent.init_agent()
async_client = AsyncOpenAI(api_key=Config.DEEPSEEK_API_KEY, base_url=Config.DEEPSEEK_BASE_URL)

@app.get("/")
async def root():
    return {"status": "ok", "agent_ready": agent.agent_executor is not None}

@app.post("/chat")
async def chat(
    query: str = Body(...),
    sys_prompt: str = Body("你是一个有用的助手。"),
    history: List = Body([]),
    history_len: int = Body(1),
    temperature: float = Body(0.5),
    top_p: float = Body(0.5),
    max_tokens: int = Body(1024),
    stream: bool = Body(True)
):
    if history_len>0:
        history = history[-2*history_len:]
    messages = [{"role":"system","content":sys_prompt}] + history + [{"role":"user","content":query}]
    resp = await async_client.chat.completions.create(
        model=Config.DEEPSEEK_MODEL, messages=messages,
        max_tokens=max_tokens, temperature=temperature, top_p=top_p, stream=stream
    )
    if stream:
        async def gen():
            async for chunk in resp:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        return StreamingResponse(gen(), media_type="text/plain")
    else:
        return {"response": resp.choices[0].message.content}

@app.get("/weather")
async def weather(city: str = Query(...)):
    result = query_weather(city)
    return {"success": "无法" not in result, "message": result}

# Agent接口（修复422错误）
@app.post("/agent")
async def agent_query(data: dict = Body(...)):
    try:
        question = data.get("question")
        if not question:
            return {"answer": "缺少 'question' 字段", "steps": []}
        result = await asyncio.get_event_loop().run_in_executor(None, agent.query_with_steps, question)
        return result
    except Exception as e:
        print(f"Agent接口错误: {e}")
        return {"answer": f"后端错误: {str(e)}", "steps": []}

# 新增语录接口
@app.get("/tiangou")
def tiangou():
    return {"content": get_tiangou()}
@app.get("/qinghua")
def qinghua():
    return {"content": get_qinghua()}
@app.get("/saohua")
def saohua():
    return {"content": get_saohua()}

# 新增文本检测接口
@app.post("/text_security")
def text_security(data: dict = Body(...)):
    text = data.get("text")
    if not text:
        return {"success": False, "message": "缺少 text 参数"}
    result = text_security_check(text)
    if "检测结果:" in result:
        parts = result.split("，")
        res = parts[0].replace("检测结果:", "").strip()
        detail = parts[1].replace("详情:", "").strip() if len(parts)>1 else ""
        return {"success": True, "result": res, "detail": detail}
    else:
        return {"success": False, "message": result}

# 成语接龙接口
@app.get("/idiom/game_state")
def game_state():
    return {"idioms_count": len(idiom_game.idioms)}
@app.post("/idiom/validate")
def validate_idiom(idiom: str = Body(...), last_idiom: str = Body(None), used_idioms: List[str] = Body([])):
    ok, reason = idiom_game.validate(idiom, last_idiom, set(used_idioms))
    return {"valid": ok, "reason": reason}
@app.post("/idiom/ai_move")
def ai_move_idiom(last_char: str = Body(...), used_idioms: List[str] = Body([])):
    result = idiom_game.ai_move(last_char, set(used_idioms))
    return {"success": result is not None, "idiom": result}


# ==================== Streamlit 前端（独立文件，已包含新页面）====================
def create_streamlit_app():
    code = '''import streamlit as st
import requests
import random

BACKEND = "http://localhost:6066"

st.set_page_config(page_title="智能应用平台", layout="wide")
mode = st.sidebar.radio("功能", ["聊天", "成语接龙", "天气", "Agent", "经典语录", "文本检测"])

# ---------- 聊天 ----------
if mode == "聊天":
    st.title("聊天机器人")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if p := st.chat_input("说点什么"):
        st.session_state.chat.append({"role": "user", "content": p})
        with st.chat_message("user"):
            st.markdown(p)
        data = {"query": p, "sys_prompt": "You are a helpful assistant.", "history": st.session_state.chat[:-1], "history_len": 2}
        r = requests.post(f"{BACKEND}/chat", json=data, stream=True)
        if r.status_code == 200:
            full = ""
            placeholder = st.chat_message("assistant")
            text = placeholder.markdown("")
            for chunk in r.iter_content(decode_unicode=True):
                full += chunk
                text.markdown(full)
            st.session_state.chat.append({"role": "assistant", "content": full})

# ---------- 天气 ----------
elif mode == "天气":
    st.title("天气查询")
    city = st.text_input("城市")
    if st.button("查询") and city:
        r = requests.get(f"{BACKEND}/weather", params={"city": city})
        if r.status_code == 200:
            data = r.json()
            if data["success"]:
                st.success(data["message"])
            else:
                st.error(data["message"])

# ---------- 成语接龙 ----------
elif mode == "成语接龙":
    st.title("成语接龙")
    if "g" not in st.session_state:
        st.session_state.g = {"used": [], "last": None, "turn": None, "over": False}
    if st.session_state.g["turn"] is None:
        r = requests.get(f"{BACKEND}/idiom/game_state")
        if r.status_code == 200:
            st.session_state.g["turn"] = random.choice(["player", "ai"])
            st.rerun()
    if st.session_state.g["over"]:
        st.success("游戏结束")
        if st.button("新游戏"):
            st.session_state.g = {"used": [], "last": None, "turn": random.choice(["player", "ai"]), "over": False}
            st.rerun()
    else:
        st.info(f"回合: {'玩家' if st.session_state.g['turn']=='player' else 'AI'}")
        if st.session_state.g["last"]:
            st.info(f"上一个: {st.session_state.g['last']}  需要以 '{st.session_state.g['last'][-1]}' 开头")
        if st.session_state.g["turn"] == "player":
            w = st.text_input("成语")
            if st.button("提交") and w:
                resp = requests.post(f"{BACKEND}/idiom/validate", json={"idiom": w, "last_idiom": st.session_state.g["last"], "used_idioms": st.session_state.g["used"]})
                if resp.status_code == 200 and resp.json()["valid"]:
                    st.session_state.g["used"].append(w)
                    st.session_state.g["last"] = w
                    ai_resp = requests.post(f"{BACKEND}/idiom/ai_move", json={"last_char": w[-1], "used_idioms": st.session_state.g["used"]})
                    if ai_resp.status_code == 200 and ai_resp.json().get("success"):
                        ai_w = ai_resp.json()["idiom"]
                        if ai_w:
                            st.session_state.g["used"].append(ai_w)
                            st.session_state.g["last"] = ai_w
                            st.rerun()
                        else:
                            st.session_state.g["over"] = True
                            st.success("AI无法接龙，你赢了！")
                            st.rerun()
                    else:
                        st.error("AI响应失败")
                else:
                    st.error(resp.json().get("reason", "无效成语"))
        else:
            st.info("AI思考中...")
            if st.button("刷新"):
                st.rerun()

# ---------- Agent ----------
elif mode == "Agent":
    st.title("Agent助手")
    st.caption("支持：天气、时间、计算器、驾考咨询、文本检测、经典语录等")
    q = st.text_input("问题", placeholder="例如：给我来一条舔狗语录")
    if st.button("运行") and q:
        with st.spinner("思考中..."):
            try:
                r = requests.post(f"{BACKEND}/agent", json={"question": q}, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    st.success(f"答案: {data['answer']}")
                    if data.get("steps"):
                        with st.expander("推理过程"):
                            for step in data["steps"]:
                                st.markdown(step)
                else:
                    st.error(f"后端错误，状态码: {r.status_code}")
            except Exception as e:
                st.error(f"请求异常: {e}")

# ---------- 经典语录独立页面 ----------
elif mode == "经典语录":
    st.title("📖 经典语录大全")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🐶 舔狗语录"):
            r = requests.get(f"{BACKEND}/tiangou")
            if r.status_code == 200:
                st.success(r.json()["content"])
            else:
                st.error("获取失败")
    with col2:
        if st.button("💖 情话语录"):
            r = requests.get(f"{BACKEND}/qinghua")
            if r.status_code == 200:
                st.success(r.json()["content"])
            else:
                st.error("获取失败")
    with col3:
        if st.button("🔥 骚话语录"):
            r = requests.get(f"{BACKEND}/saohua")
            if r.status_code == 200:
                st.success(r.json()["content"])
            else:
                st.error("获取失败")

# ---------- 文本检测独立页面 ----------
elif mode == "文本检测":
    st.title("🔍 AI文本违规检测")
    text = st.text_area("请输入要检测的文本", height=150)
    if st.button("检测安全") and text:
        r = requests.post(f"{BACKEND}/text_security", json={"text": text})
        if r.status_code == 200:
            data = r.json()
            if data["success"]:
                st.info(f"检测结果：{data['result']}")
                if data.get("detail"):
                    st.write(f"详情：{data['detail']}")
            else:
                st.error(data["message"])
        else:
            st.error("检测服务异常")
'''
    script_path = os.path.join(os.path.dirname(__file__), "streamlit_app.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)
    return script_path

def run_streamlit():
    script = create_streamlit_app()
    subprocess.Popen(["streamlit", "run", script], shell=True)

# ==================== 主程序 ====================
def start_web_service():
    def run_backend():
        uvicorn.run(app, host="0.0.0.0", port=Config.BACKEND_PORT, log_level="warning")
    t = threading.Thread(target=run_backend, daemon=True)
    t.start()
    time.sleep(1)
    print(f"✅ 后端启动: http://localhost:{Config.BACKEND_PORT}")
    run_streamlit()
    print("✅ 前端已启动，浏览器将打开 http://localhost:8501")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("关闭服务")

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║  完整AI平台 - 新增驾考、文本检测、经典语录（舔狗/情话/骚话）║
║  功能：聊天 | 成语接龙 | 天气 | Agent工具 | 独立语录页面    ║
╚══════════════════════════════════════════════════════════════╝
    """)
    print("1. 基础推理  2. 成语接龙  3. Agent命令行  4. 启动Web服务  5. 退出")
    c = input("请选择: ").strip()
    if c == "1":
        BasicInference().run_demo()
    elif c == "2":
        IdiomSolitaire().run_demo()
    elif c == "3":
        AgentDemo().run_demo()
    elif c == "4":
        start_web_service()
    else:
        print("再见")

if __name__ == "__main__":
    main()