import streamlit as st
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
