import streamlit as st
import uuid
import random
from datetime import datetime
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Tongyi
from langchain.prompts import PromptTemplate
import json
import time
import os
import glob
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.schema import Document


# ================== 页面配置 ==================
st.set_page_config(page_title="AI产品经理学习助手", layout="wide")
st.markdown("""
<style>
    /* 调整侧边栏宽度，避免“新建”按钮换行 */
    section[data-testid="stSidebar"] {
        width: 280px !important;
        min-width: 280px !important;
    }
    /* 确保按钮文字不换行 */
    .stButton button {
        white-space: nowrap;
    }
    /* 减少“会话列表”和“新建”按钮之间的距离 */
    .stSidebar .stMarkdown h1, .stSidebar .stMarkdown h2, .stSidebar .stMarkdown h3 {
        margin-bottom: 0.2rem !important;
    }
    /* 新建按钮上方的间距减半 */
    .stSidebar .stButton:first-of-type {
        margin-top: -0.8rem;
    }
    /* 调整会话列表项之间的间距 */
    .stSidebar .stButton button {
        margin-top: 0.1rem;
        margin-bottom: 0.1rem;
    }
</style>
""", unsafe_allow_html=True)
st.title("📘 AI产品经理学习助手")

# ================== 加载知识库（缓存） ==================
@st.cache_resource
def load_knowledge_base():
    loader = DirectoryLoader("./knowledge_base", glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    documents = loader.load()
    
    # 配置标题切分器（用于长文件）
    headers_to_split_on = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # 配置二次切分器（固定大小）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )
    
    all_chunks = []
    for doc in documents:
        content = doc.page_content
        # 动态阈值：如果文件内容长度 ≤ 800 字符，整个文件作为一个 chunk
        if len(content) <= 800:
            chunk = Document(page_content=content, metadata={"source": doc.metadata["source"]})
            all_chunks.append(chunk)
        else:
            # 长文件：按标题切分后，再按字符大小二次切分
            header_chunks = markdown_splitter.split_text(content)
            sub_chunks = text_splitter.split_documents(header_chunks)
            for chunk in sub_chunks:
                chunk.metadata["source"] = doc.metadata["source"]
            all_chunks.extend(sub_chunks)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 分批插入，每批 100 个 chunk
    batch_size = 100
    vectordb = None
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        if vectordb is None:
            vectordb = Chroma.from_documents(batch, embeddings, persist_directory="./chroma_db")
        else:
            vectordb.add_documents(batch)
    vectordb.persist()
    return vectordb


# 构建知识库目录树，返回节点列表（每个文件一个根节点）
@st.cache_data
def build_kb_tree():
    """构建知识库目录树，返回节点列表（每个文件一个节点）"""
    md_files = glob.glob("./knowledge_base/**/*.md", recursive=True)
    tree = []
    for file_path in md_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        file_node = {
            "title": os.path.basename(file_path).replace(".md", ""),
            "file_path": file_path,
            "full_content": content,   # 存储全文
            "level": 0,
            "children": []             # 不再使用子节点，但保留结构以兼容
        }
        tree.append(file_node)
    return tree

def generate_daily_question():
    """根据掌握度最低的10个知识点中随机抽取一个，生成每日一题"""
    mastery = st.session_state.knowledge_mastery
    tree = st.session_state.kb_tree if "kb_tree" in st.session_state else build_kb_tree()
    
    # 收集掌握度数据
    mastery_list = []
    for src, data in mastery.items():
        if data["total"] > 0:
            ratio = data["correct"] / data["total"]
            mastery_list.append((src, ratio))
    mastery_list.sort(key=lambda x: x[1])  # 掌握率升序
    
    # 选取掌握度最低的10个（或全部）
    candidates = mastery_list[:10] if len(mastery_list) >= 10 else mastery_list
    
    if candidates:
        # 随机抽取一个
        selected_source = random.choice(candidates)[0]
        # 从tree中找到对应的文件内容
        content = None
        for node in tree:
            if node["file_path"] == selected_source:
                content = node["full_content"]
                break
        if not content:
            # 降级：随机选一个文件
            selected_source = random.choice(tree)["file_path"]
            content = next(node["full_content"] for node in tree if node["file_path"] == selected_source)
    else:
        # 没有任何掌握记录，随机选一个文件
        selected_node = random.choice(tree)
        selected_source = selected_node["file_path"]
        content = selected_node["full_content"]
    
    # 调用LLM生成题目
    llm = Tongyi(model="qwen-plus", dashscope_api_key=st.secrets["DASHSCOPE_API_KEY"])
    prompt = f"""请根据以下知识内容生成一道AI产品经理相关的开放式问答题。题目应考察对概念的理解或实际应用能力。

要求：
1. 题型：问答题
2. 题目清晰，要求用户用自己的话回答。
3. 输出格式严格如下：
题目：[题目内容]
参考答案：[一份完整、准确的参考答案要点]

知识内容：
{content}

请直接输出，不要有多余解释。"""
    try:
        response = llm.invoke(prompt)
        lines = response.split('\n')
        q_text = ""
        q_answer = ""
        for line in lines:
            if line.startswith("题目："):
                q_text = line.replace("题目：", "").strip()
            elif line.startswith("参考答案："):
                q_answer = line.replace("参考答案：", "").strip()
        if not q_text:
            q_text = "生成题目失败，请重试"
        return {
            "question": q_text,
            "answer": q_answer,
            "source": selected_source,
            "date": datetime.now().date()
        }
    except Exception as e:
        st.error(f"生成每日一题失败：{e}")
        return None



# ================== 辅助函数 ==================
def save_current_session():
    """保存当前会话的消息到 sessions 字典"""
    if st.session_state.current_session_id in st.session_state.sessions:
        st.session_state.sessions[st.session_state.current_session_id]["messages"] = st.session_state.messages
        # 自动更新普通会话的标题（取第一条用户消息）
        if not st.session_state.sessions[st.session_state.current_session_id].get("is_quiz"):
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    title = msg["content"][:30] + ("..." if len(msg["content"]) > 30 else "")
                    st.session_state.sessions[st.session_state.current_session_id]["title"] = title
                    break

def switch_session(session_id):
    """切换到指定会话，不自动刷新"""
    if st.session_state.quiz_mode:
        end_quiz_and_save_title()
    if st.session_state.flashcard_mode:
        st.session_state.flashcard_mode = False
        st.session_state.flashcard_deck = []
    save_current_session()
    st.session_state.current_session_id = session_id
    st.session_state.messages = st.session_state.sessions[session_id]["messages"].copy()
    st.session_state.chain = None
    st.session_state.quiz_mode = False
    st.session_state.current_question_data = None
    st.session_state.study_mode = "问答题"
    st.session_state.show_kb_browser = False   # 新增：退出知识库模式
    st.session_state.show_learning_progress = False

def create_new_session(is_quiz=False):
    """创建新会话，返回新会话ID，不自动切换"""
    new_id = str(uuid.uuid4())
    now = datetime.now()
    if is_quiz:
        title = "测试中..."
    else:
        title = "新对话"
    st.session_state.sessions[new_id] = {
        "id": new_id,
        "title": title,
        "created_at": now,
        "messages": [],
        "is_quiz": is_quiz,
        "quiz_start_time": now if is_quiz else None,
    }
    st.session_state.show_kb_browser = False   # 新增：退出知识库模式
    st.session_state.show_learning_progress = False
    return new_id

def delete_session(session_id):
    """删除会话"""
    if session_id in st.session_state.sessions:
        del st.session_state.sessions[session_id]
    if session_id == st.session_state.current_session_id:
        if st.session_state.sessions:
            first_id = list(st.session_state.sessions.keys())[0]
            switch_session(first_id)
            st.rerun()   # 新增：刷新页面
        else:
            create_new_session()
            st.rerun()   # 新增：刷新页面
    else:
        st.rerun()

def end_test_and_save_title(test_type):
    """结束测试，更新会话标题并保存会话"""
    sid = st.session_state.current_session_id
    if sid in st.session_state.sessions:
        end_time = datetime.now()
        score = st.session_state.quiz_score
        total = st.session_state.quiz_total
        title = f"{score:.1f}/{total} {test_type} {end_time.strftime('%Y/%m/%d %H:%M')}"
        st.session_state.sessions[sid]["title"] = title
    # 保存会话消息
    save_current_session()
    # 重置测试状态
    st.session_state.quiz_mode = False
    st.session_state.quiz_question = None
    st.session_state.quiz_answer = None
    st.session_state.quiz_score = 0.0
    st.session_state.quiz_total = 0
    st.session_state.quiz_chunk = None
    st.session_state.current_question_data = None

def end_quiz_and_save_title():
    """结束当前测验，将会话标题更新为‘得分/总分 测试 结束时间’"""
    if not st.session_state.quiz_mode:
        return
    sid = st.session_state.current_session_id
    if sid in st.session_state.sessions and st.session_state.sessions[sid].get("is_quiz"):
        end_time = datetime.now()
        score = st.session_state.quiz_score
        total = st.session_state.quiz_total
        # 格式：0.6/5 测试 2026-04-05 15-30
        title = f"{score:.1f}/{total} 测试 {end_time.strftime('%Y-%m-%d %H-%M')}"
        st.session_state.sessions[sid]["title"] = title
        st.session_state.sessions[sid]["quiz_end_time"] = end_time
    # 重置测验状态
    st.session_state.quiz_mode = False
    st.session_state.quiz_question = None
    st.session_state.quiz_answer = None
    st.session_state.quiz_score = 0.0
    st.session_state.quiz_total = 0
    st.session_state.quiz_chunk = None

def exit_quiz():
    """用户主动结束测验（侧边栏按钮）"""
    end_quiz_and_save_title()
    st.rerun()

def generate_quiz_question():
    """从知识库中随机抽取一个 chunk，调用 LLM 生成一道开放式问答题"""
    if st.session_state.vectordb is None:
        st.warning("知识库未加载，请稍后再试")
        return None, None, None
    try:
        collection = st.session_state.vectordb._collection
        all_data = collection.get(include=["documents", "metadatas"])
        if not all_data or len(all_data['documents']) == 0:
            st.error("知识库为空，请先添加文档")
            return None, None, None
        idx = random.randint(0, len(all_data['documents']) - 1)
        chunk_content = all_data['documents'][idx]
        chunk_metadata = all_data['metadatas'][idx]
    except Exception as e:
        st.error(f"读取知识库失败：{e}")
        return None, None, None

    llm = Tongyi(model="qwen-turbo", dashscope_api_key=st.secrets["DASHSCOPE_API_KEY"])
    prompt = f"""请根据以下知识内容生成一道AI产品经理相关的**开放式问答题**（非选择题、非判断题）。题目应考察对概念的理解或实际应用能力。

要求：
1. 题型：问答题
2. 题目清晰，要求用户用自己的话回答。
3. 输出格式严格如下：
题型：问答题
题目：[题目内容]
参考答案：[一份完整、准确的参考答案要点，用于评分]

知识内容：
{chunk_content}

请直接输出，不要有多余解释。"""
    try:
        response = llm.invoke(prompt)
        lines = response.split('\n')
        q_text = ""
        q_answer = ""
        for line in lines:
            if line.startswith("题目："):
                q_text = line.replace("题目：", "").strip()
            elif line.startswith("参考答案："):
                q_answer = line.replace("参考答案：", "").strip()
        if not q_text:
            q_text = "生成题目失败，请重试"
        full_question = f"【问答题】\n{q_text}"
        return full_question, q_answer, chunk_metadata
    except Exception as e:
        st.error(f"生成题目失败：{e}")
        return None, None, None

def evaluate_answer(user_answer, reference_answer, question_text, chunk_metadata):
    """评分，低分时返回详细建议"""
    llm = Tongyi(model="qwen-turbo", dashscope_api_key=st.secrets["DASHSCOPE_API_KEY"])
    prompt_simple = f"""你是AI产品经理学习助手的阅卷老师。请对用户的回答进行评分。

题目：{question_text}
参考答案（要点）：{reference_answer}
用户回答：{user_answer}

请严格按照以下格式输出（每行一个项目）：
准确率：[0-100]%
错误点：[指出用户回答中缺少或错误的部分，如果没有则写“无”]
修正建议：[给出如何改进回答的具体建议，可以引用参考答案中的要点]

注意：
- 准确率基于用户回答覆盖了参考答案要点的百分比，要客观。
- 错误点要具体，不要说“回答不完整”，而要说“缺少了对RICE模型中Reach的定义”等。
- 修正建议要可操作。
"""
    try:
        response = llm.invoke(prompt_simple)
        accuracy = 0
        error_points = ""
        suggestions = ""
        for line in response.split('\n'):
            if line.startswith("准确率："):
                acc_str = line.replace("准确率：", "").strip().replace("%", "")
                try:
                    accuracy = int(acc_str)
                except:
                    accuracy = 0
            elif line.startswith("错误点："):
                error_points = line.replace("错误点：", "").strip()
            elif line.startswith("修正建议："):
                suggestions = line.replace("修正建议：", "").strip()
        if not error_points:
            error_points = "无"
        if not suggestions:
            suggestions = "请参考参考答案完善回答。"
    except Exception as e:
        st.error(f"评分失败：{e}")
        return 0, "评分失败", "请重试", ""

    detailed_review = ""
    if accuracy < 85:
        prompt_detailed = f"""你是AI产品经理学习助手的辅导老师。用户对以下问题的回答准确率较低（{accuracy}%），请为他提供一份详细的复习材料。

题目：{question_text}
参考答案（要点）：{reference_answer}
用户回答：{user_answer}
错误点：{error_points}

请输出一份【详细建议】，包含以下三个部分，每个部分的标题用Markdown加粗（即**标题**）：

**1. 核心知识点讲解**
（展开参考答案中的每个要点，用通俗语言解释）

**2. 标准答案范例**
（完整、有条理的答案）

**3. 用户回答与标准答案的差距分析**
（具体指出缺失了哪些关键点）

要求：内容详实，不少于300字，帮助用户真正理解这个知识点。
直接输出“详细建议：”开头的内容，不要有其他格式。
"""
        try:
            detailed_response = llm.invoke(prompt_detailed)
            detailed_review = detailed_response.strip()
        except Exception as e:
            detailed_review = f"生成详细建议失败：{e}"

    return accuracy, error_points, suggestions, detailed_review

def evaluate_scenario(user_answer, scenario_text):
    """评估场景模拟回答，返回准确率、错误点、修正建议、详细建议"""
    llm = Tongyi(model="qwen-max", dashscope_api_key=st.secrets["DASHSCOPE_API_KEY"])
    prompt = f"""你是一位产品专家。请评估用户对以下业务场景的分析框架和验证方案。

场景描述：{scenario_text}
用户回答：{user_answer}

请按照以下格式输出：
准确率：[0-100]%
错误点：[指出用户回答中缺少或错误的部分]
修正建议：[给出改进建议]

注意：准确率基于分析框架的完整性、逻辑性和可操作性。"""
    try:
        response = llm.invoke(prompt)
        # 解析准确率等
        accuracy = 0
        error_points = ""
        suggestions = ""
        for line in response.split('\n'):
            if line.startswith("准确率："):
                acc_str = line.replace("准确率：", "").strip().replace("%", "")
                try:
                    accuracy = int(acc_str)
                except:
                    accuracy = 0
            elif line.startswith("错误点："):
                error_points = line.replace("错误点：", "").strip()
            elif line.startswith("修正建议："):
                suggestions = line.replace("修正建议：", "").strip()
        if not error_points:
            error_points = "无"
        if not suggestions:
            suggestions = "请参考优秀案例分析框架。"
        detailed_review = ""
        if accuracy < 85:
            detailed_prompt = f"""用户对以下场景的回答准确率较低（{accuracy}%），请提供详细建议。

场景：{scenario_text}
用户回答：{user_answer}
错误点：{error_points}

请按照以下格式输出：

**一、关键要点**
（列出2-3个最重要的知识点）

**二、标准分析框架示例：问题诊断五步法**
（给出五步法的具体步骤）

**三、5W1H分析**（以表格形式，只输出表格）
| What | Why | When | Where | Who | How |
|------|-----|------|-------|-----|-----|
| ... | ... | ... | ... | ... | ... |

**四、参考答案**
（严格遵守150-200字）

不需要其他内容。"""
            detailed_response = llm.invoke(detailed_prompt)
            detailed_review = detailed_response.strip()
        return accuracy, error_points, suggestions, detailed_review
    except Exception as e:
        return 0, "评估失败", str(e), ""

def process_mock_interview(user_answer):
    """处理模拟面试的回答（包括自动超时）"""
    # 标记已回答（防止重复触发超时）
    st.session_state.current_question_data["answered"] = True
    
    # 创建 LLM 实例
    llm = Tongyi(model="qwen-turbo", dashscope_api_key=st.secrets["DASHSCOPE_API_KEY"])
    
    current_idx = st.session_state.interview_current_index
    st.session_state.interview_responses.append(user_answer)
    
    # 计算是否超时
    elapsed = time.time() - st.session_state.interview_start_time
    response_time = elapsed  # 记录回答用时
    feedback = ""
    if elapsed > 90:
        feedback += f"\n\n⏰ 超时提醒：你的回答用时 {elapsed:.0f} 秒，超过90秒。建议下次加快思考。"
    
    # 判断是否还有下一题
    if current_idx + 1 < st.session_state.interview_total:
        # 先处理超时提醒（如果有）
        if elapsed > 90:
            timeout_msg = f"⏰ 超时提醒：你的回答用时 {elapsed:.0f} 秒，超过90秒。建议下次加快思考。"
            with st.chat_message("assistant"):
                st.markdown(timeout_msg)
            st.session_state.messages.append({"role": "assistant", "content": timeout_msg})
            save_current_session()
        # 进入下一题
        st.session_state.interview_current_index += 1
        st.session_state.interview_start_time = time.time()
        next_q = st.session_state.interview_questions[current_idx+1].replace('Q:', '面试官：').strip()
        timeout_seconds = 90
        feedback = f"回答已记录（用时 {response_time:.1f} 秒）。\n\n{next_q}\n\n⏱️ **限时 90 秒**，计时开始。"
        st.session_state.current_question_data["question"] = next_q
        st.session_state.current_question_data["question_time"] = datetime.now()
        # 显示下一题信息
        with st.chat_message("assistant"):
            st.markdown(feedback)
        st.session_state.messages.append({"role": "assistant", "content": feedback})
        save_current_session()
        st.rerun()
    else:
        # 先显示回答用时
        time_msg = f"回答已记录（用时 {response_time:.1f} 秒）。"
        with st.chat_message("assistant"):
            st.markdown(time_msg)
        st.session_state.messages.append({"role": "assistant", "content": time_msg})
        save_current_session()
        
        # 准备综合评分
        qa_pairs = ""
        for i, q in enumerate(st.session_state.interview_questions):
            q_text = q.replace('Q:', '').strip()
            user_resp = st.session_state.interview_responses[i] if i < len(st.session_state.interview_responses) else "无回答"
            qa_pairs += f"问题{i+1}: {q_text}\n用户回答: {user_resp}\n\n"
        llm_max = Tongyi(model="qwen-max", dashscope_api_key=st.secrets["DASHSCOPE_API_KEY"])
        score_prompt = f"""你是一位资深AI产品经理面试官。请对以下面试问答进行综合评分（0-100分），并给出详细评价。
{qa_pairs}
输出格式：
准确率：[0-100]%
错误点：[逐题指出主要问题]
修正建议：[针对每个问题的改进建议]
总结评价：[一句话总结]
"""
        with st.spinner("正在综合评分中..."):
            score_response = llm_max.invoke(score_prompt)
            # 提取准确率
            import re
            accuracy_match = re.search(r'准确率：(\d+)', score_response)
            accuracy = int(accuracy_match.group(1)) if accuracy_match else 0
            # 移除准确率行，保留其余内容
            lines = score_response.split('\n')
            filtered_lines = [line for line in lines if not (line.startswith('准确率：') or line.startswith('输出格式：'))]
            filtered_response = '\n'.join(filtered_lines)
            # 加粗标题
            filtered_response = filtered_response.replace('错误点：', '**错误点：**')
            filtered_response = filtered_response.replace('修正建议：', '**修正建议：**')
            filtered_response = filtered_response.replace('总结评价：', '**总结评价：**')
            feedback = filtered_response
            
            # 记录得分（用于标题，但不显示）
            st.session_state.quiz_score = accuracy / 100.0 * st.session_state.interview_total
            st.session_state.quiz_total = st.session_state.interview_total
            
            # 如果准确率低于85%，额外生成参考答案
            if accuracy < 85:
                ref_prompt = f"""请为以下面试题提供参考答案（适合面试回答，每条答案150-200字，表述清晰、有条理）：
{qa_pairs}
请按照“问题1：”、“问题2：”、“问题3：”的格式输出，问题编号加粗。"""
                with st.spinner("生成参考答案中..."):
                    ref_response = llm_max.invoke(ref_prompt)
                feedback += f"\n\n📝 **参考答案**\n{ref_response}"
            
            # 结束测试，更新标题
            end_test_and_save_title("模拟面试")
            st.session_state.current_question_data = None
            st.session_state.quiz_mode = False
            # 重置模拟面试状态
            st.session_state.interview_questions = []
            st.session_state.interview_responses = []
            st.session_state.interview_current_index = 0
            if "interview_start_time" in st.session_state:
                del st.session_state.interview_start_time
        
        # 显示最终反馈
        with st.chat_message("assistant"):
            st.markdown(feedback)
        st.session_state.messages.append({"role": "assistant", "content": feedback})
        save_current_session()
        st.rerun()

#生成 Flashcard 的函数
def generate_flashcard_deck(num_cards=15, difficulty="简单"):
    """从知识库中随机抽取 num_cards 个 chunk，调用 LLM 批量生成卡片对，难度可选"""
    if st.session_state.vectordb is None:
        st.warning("知识库未加载，请稍后再试")
        return []
    try:
        collection = st.session_state.vectordb._collection
        all_data = collection.get(include=["documents", "metadatas"])
        if not all_data or len(all_data['documents']) == 0:
            st.error("知识库为空，请先添加文档")
            return []
        total_chunks = len(all_data['documents'])
        indices = random.sample(range(total_chunks), min(num_cards, total_chunks))
        chunks_content = [all_data['documents'][i] for i in indices]
    except Exception as e:
        st.error(f"读取知识库失败：{e}")
        return []

    # 根据难度定义不同的 prompt 要求
    difficulty_prompts = {
        "简单": """
- 正面：提出一个清晰的问题或关键词（例如：“什么是RICE模型？”）。
- 背面：给出简洁、准确的答案或解释（100字以内，突出核心要点）。
""",
        "中等": """
- 正面：提出一个需要简要分析或举例的问题（例如：“如何应用RICE模型给功能排优先级？”）。
- 背面：给出分点列举的答案，包含关键步骤或使用场景，字数控制在150-200字。
""",
        "困难": """
- 正面：提出一个需要深度分析、跨概念对比或解决实际问题的问题（例如：“当你的产品同时有四个功能待开发，如何结合RICE和Kano模型进行决策？请说明步骤。”）。
- 背面：给出结构化的详细解答，包括：思考框架、具体计算/评估方法、常见陷阱、实际案例，字数在250-300字。
"""
    }
    requirement = difficulty_prompts.get(difficulty, difficulty_prompts["简单"])

    # 根据难度选择模型
    if difficulty == "困难":
        llm = Tongyi(model="qwen-plus", dashscope_api_key=st.secrets["DASHSCOPE_API_KEY"])
    else:
        llm = Tongyi(model="qwen-turbo", dashscope_api_key=st.secrets["DASHSCOPE_API_KEY"])
    
    chunks_text = "\n\n---\n\n".join([f"【知识点 {i+1}】\n{chunk}" for i, chunk in enumerate(chunks_content)])
    prompt = f"""请根据以下 {num_cards} 个知识内容，为每个内容生成一张抽认卡（Flashcard），用于AI产品经理学习。

难度级别：{difficulty}
具体要求：
{requirement}

输出格式必须为严格的 JSON 数组，每个元素包含 "front" 和 "back" 字段，例如：
[
  {{"front": "问题1", "back": "答案1"}},
  {{"front": "问题2", "back": "答案2"}}
]

知识内容：
{chunks_text}

请直接输出 JSON 数组，不要有其他任何解释或额外文字。"""
    try:
        response = llm.invoke(prompt)
        json_str = response.strip()
        start = json_str.find('[')
        end = json_str.rfind(']')
        if start != -1 and end != -1:
            json_str = json_str[start:end+1]
            deck = json.loads(json_str)
            if isinstance(deck, list) and all('front' in c and 'back' in c for c in deck):
                return deck
        st.error("卡片生成格式错误，请重试")
        return []
    except Exception as e:
        st.error(f"生成卡片失败：{e}")
        return []

def generate_study_question(mode):
    """根据学习模式生成题目，返回 (question_text, extra_data)"""
    llm = Tongyi(model="qwen-turbo", dashscope_api_key=st.secrets["DASHSCOPE_API_KEY"])
    if mode == "问答题":
        # 复用原有的 generate_quiz_question 逻辑，但只返回题目文本和参考答案
        # 为了简化，直接调用原来的函数，但需要适配返回值
        full_question, ref_answer, chunk_meta = generate_quiz_question()
        if full_question:
            return full_question, {"reference_answer": ref_answer, "chunk_metadata": chunk_meta}
        else:
            return None, None
    elif mode == "模拟面试":
        llm_plus = Tongyi(model="qwen-plus", dashscope_api_key=st.secrets["DASHSCOPE_API_KEY"])
        total = st.session_state.interview_total
        prompt = f"请生成 {total} 道针对 AI产品经理的面试题，要求考察产品思维和 AI 技术理解。每道题单独一行，以'Q:'开头。"
        response = llm_plus.invoke(prompt)
        questions = [q.strip() for q in response.split('\n') if q.strip().startswith('Q:')]
        if len(questions) < total:
            # 降级使用默认问题
            questions = [
                "Q: 请解释什么是RAG（检索增强生成）？它解决了什么问题？",
                "Q: 如何为一个AI功能定义成功的衡量指标？请举例说明。",
                "Q: 描述一个你曾经解决过的复杂问题，并说明你在其中扮演的角色。"
            ][:total]
        st.session_state.interview_questions = questions
        st.session_state.interview_responses = []
        st.session_state.interview_current_index = 0
        first_question = questions[0].replace('Q:', '面试官：').strip()
        # 添加限时提示
        first_question += f"\n\n⏱️ **限时 90 秒**，计时开始。"
        return first_question, {"questions": questions, "current": 0}
    elif mode == "场景模拟":
        max_attempts = 5
        for attempt in range(max_attempts):
            llm_plus = Tongyi(model="qwen-plus", dashscope_api_key=st.secrets["DASHSCOPE_API_KEY"])
            # 构建多样性 prompt
            diversity_prompt = """请生成一个AI产品经理工作中可能遇到的模糊业务场景，要求：
1. 每次生成的场景必须具有明显的差异性，避免重复。
2. 从以下维度中随机组合：行业（电商、金融、医疗、教育、社交、物流等）、问题类型（用户增长、转化率、留存、功能使用、成本、效率等）、指标（日活、月活、GMV、CTR、CAC、LTV等）。
3. 场景描述应包含具体的数据变化（如“下降X%”）和模糊的信息缺口（如“数据不完整”）。
4. 只输出场景描述，不超过100字，不要有任何额外解释。

示例（仅供参考，不要照抄）：
- 某社交APP新功能上线后，次日留存率下降8%，但后台数据显示用户点击率正常，无法判断是功能设计问题还是内容质量问题。
- 电商大促期间，支付成功率从98%降至92%，技术排查未发现明显故障，怀疑是第三方支付通道波动，但缺乏详细日志。

请生成一个全新的场景："""
            scenario = llm_plus.invoke(diversity_prompt).strip()
            # 检查是否重复（简单比较字符串相似度，这里用完全匹配）
            if scenario not in st.session_state.used_scenarios:
                st.session_state.used_scenarios.append(scenario)
                question = f"【场景模拟】\n{scenario}\n\n请设计你的分析框架和验证方案。"
                return question, {"scenario": scenario}
        # 如果多次尝试后仍重复，使用最后一个
        st.session_state.used_scenarios.append(scenario)
        question = f"【场景模拟】\n{scenario}\n\n请设计你的分析框架和验证方案。"
        return question, {"scenario": scenario}
    else:
        return None, None

# ================== 自定义 Prompt ==================
prompt_template = """你是一个AI产品经理学习助手，专门帮助用户学习AI产品经理的知识和技能。

【核心规则】
1. 必须严格基于以下"知识库内容"来回答问题。
2. 如果知识库中没有明确提到相关信息，请直接回答："根据现有资料，我无法找到这个问题的答案。"
3. 不要编造、不要猜测、不要使用你自己的知识库之外的信息。
4. 回答要简洁、准确，用中文回答。
5. 如果可能，在回答末尾注明参考的知识来源。

【知识库内容】
{context}

【对话历史】
{chat_history}

【用户问题】
{question}

【回答】
"""
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "chat_history", "question"]
)

# ================== 获取对话链 ==================
def get_conversation_chain(vectordb):
    llm = Tongyi(model="qwen-turbo", dashscope_api_key=st.secrets["DASHSCOPE_API_KEY"])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key='answer',
        combine_docs_chain_kwargs={"prompt": PROMPT},
        verbose=False
    )
    return chain, memory

# ================== 初始化会话管理 ==================
if "sessions" not in st.session_state:
    st.session_state.sessions = {}
if "current_session_id" not in st.session_state:
    # 创建一个默认普通会话
    default_id = str(uuid.uuid4())
    st.session_state.sessions[default_id] = {
        "id": default_id,
        "title": "新对话",
        "created_at": datetime.now(),
        "messages": [],
        "is_quiz": False,
        "quiz_start_time": None,
    }
    st.session_state.current_session_id = default_id
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

# 测验模式状态
if "quiz_mode" not in st.session_state:
    st.session_state.quiz_mode = False
if "quiz_question" not in st.session_state:
    st.session_state.quiz_question = None
if "quiz_answer" not in st.session_state:
    st.session_state.quiz_answer = None
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0.0
if "quiz_total" not in st.session_state:
    st.session_state.quiz_total = 0
if "quiz_chunk" not in st.session_state:
    st.session_state.quiz_chunk = None

# 学习模式（取代原来的 quiz_mode，但为了兼容，保留 quiz_mode 作为子状态）
if "study_mode" not in st.session_state:
    st.session_state.study_mode = "问答题"  # 可选: 问答题, 对比学习, 模拟面试, 场景模拟
if "current_question_data" not in st.session_state:
    st.session_state.current_question_data = None  # 存储当前题目、参考答案、额外信息等

# 模拟面试专用状态
if "interview_questions" not in st.session_state:
    st.session_state.interview_questions = []      # 存储所有面试题
if "interview_responses" not in st.session_state:
    st.session_state.interview_responses = []      # 存储用户所有回答
if "interview_current_index" not in st.session_state:
    st.session_state.interview_current_index = 0
if "interview_total" not in st.session_state:
    st.session_state.interview_total = 3            # 默认3个问题

# 场景模拟状态
if "used_scenarios" not in st.session_state:
    st.session_state.used_scenarios = []

# Flashcard 模式状态
if "flashcard_mode" not in st.session_state:
    st.session_state.flashcard_mode = False
if "flashcard_deck" not in st.session_state:
    st.session_state.flashcard_deck = []
if "flashcard_index" not in st.session_state:
    st.session_state.flashcard_index = 0
if "flashcard_show_back" not in st.session_state:
    st.session_state.flashcard_show_back = False
if "flashcard_refresh_flag" not in st.session_state:
    st.session_state.flashcard_refresh_flag = 0   # 可选，用于强制刷新标识
# 每张卡片的掌握状态（True=已掌握，False=未掌握）
if "flashcard_mastered" not in st.session_state:
    st.session_state.flashcard_mastered = []  # 与 flashcard_deck 同步长度的 bool 列表
if "flashcard_stats" not in st.session_state:
    st.session_state.flashcard_stats = {}  # {卡片内容: {"total": int, "mastered": int}}

# 知识库浏览器显示状态
if "show_kb_browser" not in st.session_state:
    st.session_state.show_kb_browser = False
# 掌握度跟踪
if "knowledge_mastery" not in st.session_state:
    st.session_state.knowledge_mastery = {}  # {source: {"correct": int, "total": int}}
# 学习进度浏览器显示状态
if "show_learning_progress" not in st.session_state:
    st.session_state.show_learning_progress = False

# 每日一题相关状态
if "last_visit_date" not in st.session_state:
    st.session_state.last_visit_date = None
if "daily_question" not in st.session_state:
    st.session_state.daily_question = None  # {"question": str, "answer": str, "source": str, "date": date}
if "daily_question_history" not in st.session_state:
    st.session_state.daily_question_history = []  # list of {"date": date, "question": str, "user_answer": str, "accuracy": int, "source": str}
if "daily_question_answered" not in st.session_state:
    st.session_state.daily_question_answered = False

# ================== 侧边栏 ==================
with st.sidebar:
    # ================== 1. 知识库按钮 ==================
    if st.button("📚 知识库", use_container_width=True):
        st.session_state.show_kb_browser = True
        st.session_state.show_learning_progress = False  # 互斥
        st.rerun()
    # ================== 2. 学习进度按钮 ==================
    if st.button("📊 学习进度", use_container_width=True):
        st.session_state.show_learning_progress = True
        st.session_state.show_kb_browser = False  # 互斥
        st.rerun()
    st.divider()
    
    # ================== 2. Flashcard 区域 ==================
    flashcard_difficulty = st.radio(
        "卡片难度",
        options=["简单", "中等", "困难"],
        index=0,
        horizontal=True,
        key="flashcard_difficulty"
    )
    if st.button("📇 生成 Flashcard", use_container_width=True):
        if st.session_state.quiz_mode:
            end_quiz_and_save_title()
        if st.session_state.flashcard_mode:
            st.session_state.flashcard_refresh_flag += 1
        else:
            st.session_state.flashcard_mode = True
        with st.spinner("生成抽认卡中（约15秒）..."):
            if st.session_state.vectordb is None:
                st.session_state.vectordb = load_knowledge_base()
            deck = generate_flashcard_deck(15, difficulty=flashcard_difficulty)
            if deck:
                st.session_state.flashcard_deck = deck
                st.session_state.flashcard_mastered = [False] * len(deck)
                st.session_state.flashcard_index = 0
                st.session_state.flashcard_show_back = False
            else:
                st.error("生成失败，请重试")
                st.session_state.flashcard_mode = False
        st.rerun()
    st.divider()
    
    # ================== 3. 测试区域 ==================
    def on_study_mode_change():
        """题型改变时，如果有未结束的测试，自动停止"""
        if st.session_state.current_question_data is not None:
            mode = st.session_state.study_mode
            if mode == "问答题":
                end_test_and_save_title("问答题")
            elif mode == "模拟面试":
                end_test_and_save_title("模拟面试")
            elif mode == "场景模拟":
                end_test_and_save_title("场景模拟")
            st.session_state.current_question_data = None
            st.session_state.messages.append({"role": "assistant", "content": "测试已自动停止，因为切换了题型。"})
            save_current_session()
            st.rerun()

    study_mode = st.radio(
        "选择测试题型",
        options=["问答题", "模拟面试", "场景模拟"],
        index=0,
        horizontal=True,
        key="study_mode_select",
        on_change=on_study_mode_change
    )
    if st.button("🚀 开始测试", use_container_width=True):
        # 保存模式到 session_state
        st.session_state.study_mode = study_mode
        # 退出其他模式（Flashcard 等）
        if st.session_state.flashcard_mode:
            st.session_state.flashcard_mode = False
            st.session_state.flashcard_deck = []
        if st.session_state.quiz_mode:
            end_quiz_and_save_title()
        # 创建新会话（测验会话，避免自动改名）
        new_id = create_new_session(is_quiz=True)
        save_current_session()
        st.session_state.current_session_id = new_id
        st.session_state.messages = []
        st.session_state.chain = None
        st.session_state.quiz_mode = False
        st.session_state.used_scenarios = []
        if st.session_state.vectordb is None:
            with st.spinner("加载知识库中..."):
                st.session_state.vectordb = load_knowledge_base()
        st.session_state.quiz_score = 0.0
        st.session_state.quiz_total = 0
        question_text, extra_data = generate_study_question(st.session_state.study_mode)
        if question_text:
            st.session_state.current_question_data = {
                "question": question_text,
                "extra": extra_data,
                "question_time": datetime.now(),
                "answered": False,
                "hint_generated": False
            }
            # 如果是模拟面试，初始化计时开始时间
            if st.session_state.study_mode == "模拟面试":
                st.session_state.interview_start_time = time.time()
            st.session_state.messages.append({"role": "assistant", "content": question_text})
            save_current_session()
            st.rerun()
        else:
            st.error("生成题目失败，请重试")
    # 停止测试按钮
    if st.session_state.current_question_data is not None:
        if st.button("🛑 停止测试", use_container_width=True):
            mode = st.session_state.study_mode
            if mode == "问答题":
                end_test_and_save_title("问答题")
            elif mode == "模拟面试":
                end_test_and_save_title("模拟面试")
            elif mode == "场景模拟":
                end_test_and_save_title("场景模拟")
            st.session_state.current_question_data = None
            st.session_state.messages.append({"role": "assistant", "content": "测试已手动停止。"})
            save_current_session()
            st.rerun()
    st.divider()
    
    # ================== 4. 对话历史 ==================
    # 标题与新建按钮行（增加下边距）
    col_title, col_new = st.columns([2, 1])  # 标题占更大比例，按钮右移但不过分
    with col_title:
        st.header("💬 对话历史")
    with col_new:
        # 使用 CSS 让按钮靠左一些（相对于列右侧）
        st.markdown('<div style="margin-left: -10px;">', unsafe_allow_html=True)
        if st.button("➕ 新建", key="new_session_btn"):
            new_id = create_new_session(is_quiz=False)
            switch_session(new_id)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # 会话列表（添加上边距）
    st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
    sorted_sessions = sorted(st.session_state.sessions.items(), key=lambda x: x[1]["created_at"], reverse=True)
    for sid, sess in sorted_sessions:
        cols = st.columns([4, 1])
        with cols[0]:
            if sid == st.session_state.current_session_id:
                st.markdown(f"**✅ {sess['title']}**")
            else:
                if st.button(sess['title'], key=f"btn_{sid}", use_container_width=True):
                    switch_session(sid)
                    st.rerun()
        with cols[1]:
            if st.button("🗑️", key=f"del_{sid}"):
                delete_session(sid)

    # 重命名会话（添加上边距）
    st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
    with st.expander("✏️ 重命名会话"):
        sess_options = list(st.session_state.sessions.keys())
        if sess_options:
            selected_sid = st.selectbox(
                "选择会话",
                options=sess_options,
                format_func=lambda x: st.session_state.sessions[x]['title'],
                key="rename_select"
            )
            new_title = st.text_input("新标题", value=st.session_state.sessions[selected_sid]['title'] if selected_sid else "", key="rename_input")
            if st.button("保存重命名", key="rename_save"):
                if selected_sid and new_title.strip():
                    st.session_state.sessions[selected_sid]['title'] = new_title.strip()
                    st.rerun()
    st.caption(f"共 {len(st.session_state.sessions)} 个会话")

# ================== 每日一题初始化 ==================
today = datetime.now().date()
if st.session_state.last_visit_date != today:
    st.session_state.last_visit_date = today
    st.session_state.daily_question = generate_daily_question()
    st.session_state.daily_question_answered = False
    # 可选：清空之前的题目，保留历史

# ================== 主聊天界面 ==================
if st.session_state.flashcard_mode:
    deck = st.session_state.flashcard_deck
    if not deck:
        st.info("暂无卡片，请点击侧边栏「生成 Flashcard」按钮。")
    else:
        idx = st.session_state.flashcard_index
        total = len(deck)
        card = deck[idx]
        # 记录学习次数（保持不变）
        if "last_counted_card" not in st.session_state:
            st.session_state.last_counted_card = None
        if st.session_state.last_counted_card != idx:
            st.session_state.last_counted_card = idx
            card_key = card["front"][:50]
            if card_key not in st.session_state.flashcard_stats:
                st.session_state.flashcard_stats[card_key] = {"total": 0, "mastered": 0}
            st.session_state.flashcard_stats[card_key]["total"] += 1

        # 两列布局：卡片 + 按钮区
        col_card, col_buttons = st.columns([3, 1])
        with col_card:
            # 卡片样式和内容（不再显示序号）
            st.markdown("""
            <style>
            .flashcard {
                border: 1px solid #ddd;
                border-radius: 20px;
                padding: 2rem;
                margin: 1rem 0;
                background-color: #f9f9ff;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                min-height: 300px;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
            }
            </style>
            """, unsafe_allow_html=True)
            if st.session_state.flashcard_show_back:
                content = card["back"]
                label = "💡 答案"
            else:
                content = card["front"]
                label = "❓ 问题"
            st.markdown(f'<div class="flashcard"><div><strong>{label}</strong><br><br>{content}</div></div>', unsafe_allow_html=True)

        with col_buttons:
            # 显示当前卡片序号和总数
            st.markdown(f"#### 当前进度 {idx+1} / {total}")
            # st.markdown("#### 操作")
            if st.button("⬅️ 上一张", disabled=(idx == 0), use_container_width=True):
                st.session_state.flashcard_index -= 1
                st.session_state.flashcard_show_back = False
                st.rerun()
            if st.button("🔄 显示答案" if not st.session_state.flashcard_show_back else "🙈 隐藏答案", use_container_width=True):
                st.session_state.flashcard_show_back = not st.session_state.flashcard_show_back
                st.rerun()
            if st.button("✅ 已掌握", key="master", use_container_width=True):
                st.session_state.flashcard_mastered[idx] = True
                card_key = card["front"][:50]
                if card_key in st.session_state.flashcard_stats:
                    st.session_state.flashcard_stats[card_key]["mastered"] += 1
                if idx + 1 < total:
                    st.session_state.flashcard_index += 1
                    st.session_state.flashcard_show_back = False
                st.rerun()
            if st.button("⭕️ 未掌握", key="unmaster", use_container_width=True):
                st.session_state.flashcard_mastered[idx] = False
                # 将当前卡片复制一份追加到 deck 末尾（未掌握复习）
                st.session_state.flashcard_deck.append(st.session_state.flashcard_deck[idx])
                st.session_state.flashcard_mastered.append(False)
                # 翻到下一张（如果存在）
                if idx + 1 < total:
                    st.session_state.flashcard_index += 1
                    st.session_state.flashcard_show_back = False
                else:
                    # 如果已经是最后一张，复制后总张数增加，当前索引不变（仍指向原来的最后一张）
                    # 但用户期望翻到下一张（即新复制的卡片），所以需要将索引指向新卡片
                    # 由于复制后总张数变为 total+1，原来的最后一张索引为 total-1，新卡片索引为 total
                    # 此时应该将索引设为 total（新卡片），并重置答案显示
                    st.session_state.flashcard_index = total
                    st.session_state.flashcard_show_back = False
                st.rerun()
            if st.button("🚪 退出卡片", use_container_width=True):
                # 生成报告并退出（复用侧边栏退出逻辑）
                if st.session_state.flashcard_stats:
                    report = "### Flashcard 学习报告\n\n"
                    report += "| 知识点 | 答案 | 学习次数 | 掌握次数 | 掌握率 |\n"
                    report += "|--------|------|----------|----------|--------|\n"
                    for card_key, stats in st.session_state.flashcard_stats.items():
                        back_content = ""
                        for card_in in st.session_state.flashcard_deck:
                            if card_in["front"][:50] == card_key:
                                back_content = card_in["back"]  # 完整内容
                                break
                        mastered_rate = stats["mastered"] / stats["total"] if stats["total"] > 0 else 0
                        report += f"| {card_key} | {back_content} | {stats['total']} | {stats['mastered']} | {mastered_rate:.0%} |\n"
                    new_id = create_new_session(is_quiz=False)
                    st.session_state.sessions[new_id]["messages"] = [{"role": "assistant", "content": report}]
                    end_time = datetime.now()
                    difficulty = st.session_state.flashcard_difficulty
                    title = f"{difficulty}Flashcard {end_time.strftime('%Y/%m/%d %H:%M')}"
                    st.session_state.sessions[new_id]["title"] = title
                    save_current_session()
                    st.session_state.current_session_id = new_id
                    st.session_state.messages = st.session_state.sessions[new_id]["messages"].copy()
                st.session_state.flashcard_mode = False
                st.session_state.flashcard_deck = []
                st.session_state.flashcard_stats = {}
                st.rerun()
        st.caption("提示：点击已掌握或未掌握会自动翻到下一张。")
else:
    # 知识库浏览器模式
    if st.session_state.show_kb_browser:
        # 显示返回按钮
        if st.button("← 返回聊天", use_container_width=False):
            st.session_state.show_kb_browser = False
            st.session_state.show_learning_progress = False
            st.session_state.weak_preview = None
            st.rerun()
        
        st.markdown("## 📚 知识库目录")
        
        search_term = st.text_input("🔍 搜索文件", placeholder="输入文件名关键词...")
        
        if "kb_tree" not in st.session_state:
            with st.spinner("加载知识库目录..."):
                st.session_state.kb_tree = build_kb_tree()
        tree = st.session_state.kb_tree
        
        if search_term:
            filtered_files = [node for node in tree if search_term.lower() in node["title"].lower()]
        else:
            filtered_files = tree
        
        if not filtered_files:
            st.info("未找到匹配的文件")
        else:
            for node in filtered_files:
                with st.expander(f"📄 {node['title']}", expanded=False):
                    st.markdown(node['full_content'])
                    if st.button("📋 复制全文", key=f"copy_{node['title']}"):
                        st.toast("请手动复制（Streamlit 暂不支持直接复制）", icon="ℹ️")
                        st.code(node['full_content'], language="markdown")
        
        st.stop()
    
    # 学习进度浏览器模式的返回按钮
    if st.session_state.show_learning_progress:
        if st.button("← 返回聊天", use_container_width=False):
            st.session_state.show_learning_progress = False
            st.session_state.show_kb_browser = False
            st.session_state.weak_preview = None
            st.rerun()
        
        # 每日一题历史记录
        st.markdown("### 📅 每日一题记录")
        with st.expander("历史记录", expanded=True):
            if not st.session_state.daily_question_history:
                st.info("暂无每日一题记录")
            else:
                table = "| 日期 | 题目摘要 | 准确率 |\n|------|----------|--------|\n"
                for record in st.session_state.daily_question_history:
                    date_str = record['date'].strftime("%Y-%m-%d")
                    question_preview = record['question'][:50] + "..."
                    accuracy = f"{record['accuracy']}%"
                    table += f"| {date_str} | {question_preview} | {accuracy} |\n"
                st.markdown(table)
                
                # 按日期降序排序，最新的在前
                sorted_records = sorted(st.session_state.daily_question_history, key=lambda x: x['date'], reverse=True)
                sorted_dates = [r['date'] for r in sorted_records]
                selected_date = st.selectbox(
                    "选择日期查看详情",
                    options=sorted_dates,
                    format_func=lambda d: d.strftime("%Y-%m-%d"),
                    index=0
                )
                if selected_date:
                    record = next(r for r in sorted_records if r['date'] == selected_date)
                    st.markdown(f"**题目**：{record['question']}")
                    st.markdown(f"**你的答案**：{record['user_answer']}")
                    st.markdown(f"**准确率**：{record['accuracy']}%")
                    st.markdown(f"**错误点**：{record.get('error_points', '无')}")
                    if record.get('detailed_review'):
                        st.markdown(f"**详细建议**：{record['detailed_review']}")
        
        mastery = st.session_state.knowledge_mastery
        if not mastery:
            st.info("暂无学习记录，开始问答题测试后会自动跟踪。")
        else:
            stats = []
            for source, data in mastery.items():
                if data["total"] == 0:
                    continue
                ratio = data["correct"] / data["total"]
                title = os.path.basename(source).replace(".md", "")
                stats.append((title, ratio, data["correct"], data["total"]))
            stats.sort(key=lambda x: x[1])  # 按掌握度升序
            
            # 掌握度最低的10个知识点（放在表格前面）
            weak_points = stats[:10]
            st.markdown("### 掌握度最低的10个知识点")
            if "weak_preview" not in st.session_state:
                st.session_state.weak_preview = None
            
            for title, ratio, correct, total in weak_points:
                col1, col2 = st.columns([4, 1])
                with col1:
                    # 按钮文本格式：掌握度 + 知识点名字
                    btn_label = f"**掌握度** {ratio:.0%}： {title}"
                    is_previewing = (st.session_state.get("weak_preview") and 
                                    st.session_state.weak_preview.get("title") == title)
                    if st.button(btn_label, key=f"weak_{title}"):
                        if is_previewing:
                            # 如果已经预览，则收起
                            st.session_state.weak_preview = None
                        else:
                            # 否则查找并设置预览
                            if "kb_tree" not in st.session_state:
                                st.session_state.kb_tree = build_kb_tree()
                            tree = st.session_state.kb_tree
                            found_content = None
                            for node in tree:
                                if node["title"] == title:
                                    found_content = node["full_content"]
                                    break
                            if found_content:
                                st.session_state.weak_preview = {
                                    "title": title,
                                    "content": found_content
                                }
                            else:
                                st.session_state.weak_preview = None
                        st.rerun()
                with col2:
                    st.write(f"{correct}/{total}")
                if st.session_state.get("weak_preview") and st.session_state.weak_preview.get("title") == title:
                    preview = st.session_state.weak_preview
                    st.markdown(f"**📖 内容预览**")
                    st.markdown(preview['content'])
                    if st.button("📋 复制全文", key=f"copy_weak_{title}"):
                        st.toast("请手动复制（Streamlit 暂不支持直接复制）", icon="ℹ️")
                        st.code(preview['content'], language="markdown")
                st.markdown("---")
            
            # 掌握度排名表格（从低到高）
            st.markdown("### 掌握度排名（从低到高）")
            table_rows = ["| 知识点 | 掌握度 | 正确次数/总次数 |"]
            table_rows.append("|--------|--------|----------------|")
            for title, ratio, correct, total in stats:
                table_rows.append(f"| {title} | {ratio:.0%} | {correct}/{total} |")
            st.markdown("\n".join(table_rows))
        
        st.stop()

    # 兜底：如果每日一题丢失但今天尚未回答，则重新生成
    if st.session_state.daily_question is None and st.session_state.last_visit_date == datetime.now().date():
        with st.spinner("正在恢复今日一题..."):
            st.session_state.daily_question = generate_daily_question()
            st.session_state.daily_question_answered = False

    # 每日一题卡片（可伸缩窗口）
    if st.session_state.daily_question is not None:
        dq = st.session_state.daily_question
        with st.expander("📅 今日一题", expanded=True):
            st.markdown(f"**题目**：{dq['question']}")
            
            if not st.session_state.daily_question_answered:
                user_answer = st.text_area("你的答案", key="daily_answer_input", height=150)
                if st.button("提交答案", key="submit_daily"):
                    if user_answer.strip():
                        accuracy, error_points, suggestions, detailed = evaluate_answer(
                            user_answer, dq['answer'], dq['question'], {"source": dq['source']}
                        )
                        st.session_state.daily_question_history.append({
                            "date": dq['date'],
                            "question": dq['question'],
                            "user_answer": user_answer,
                            "accuracy": accuracy,
                            "source": dq['source'],
                            "reference_answer": dq['answer'],
                            "error_points": error_points,
                            "detailed_review": detailed
                        })
                        source = dq['source']
                        if source not in st.session_state.knowledge_mastery:
                            st.session_state.knowledge_mastery[source] = {"correct": 0, "total": 0}
                        st.session_state.knowledge_mastery[source]["total"] += 1
                        if accuracy >= 85:
                            st.session_state.knowledge_mastery[source]["correct"] += 1
                        st.session_state.daily_question_answered = True
                        st.rerun()
                    else:
                        st.warning("请输入答案")
            else:
                today_record = next((r for r in st.session_state.daily_question_history if r['date'] == dq['date']), None)
                if today_record:
                    st.info(f"今日已作答，准确率：{today_record['accuracy']}%")
                    st.markdown(f"**你的答案**：{today_record['user_answer']}")
                    st.markdown(f"**错误点**：{today_record.get('error_points', '无')}")
                    if today_record.get('detailed_review'):
                        st.markdown(f"**详细建议**：{today_record['detailed_review']}")
                else:
                    st.warning("未找到今日记录，请尝试刷新")

    # 原有的聊天界面（显示消息、输入框）
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # 问答题提示按钮（始终显示，但根据条件禁用）
    if st.session_state.current_question_data and st.session_state.study_mode == "问答题":
        q_time = st.session_state.current_question_data.get("question_time")
        hint_generated = st.session_state.current_question_data.get("hint_generated", False)
        if q_time and not hint_generated:
            elapsed = (datetime.now() - q_time).total_seconds()
        
        col1, col2 = st.columns([1, 8])
        with col1:
            if st.button("💡 提示", key="hint_btn"):
                q_time = st.session_state.current_question_data.get("question_time")
                hint_generated = st.session_state.current_question_data.get("hint_generated", False)
                if not q_time or hint_generated:
                    st.toast("提示不可用", icon="⚠️")
                else:
                    elapsed = (datetime.now() - q_time).total_seconds()
                    if elapsed < 60:
                        st.toast(f"请等待 {int(60 - elapsed)} 秒后再获取提示", icon="⏳")
                    else:
                        st.session_state.current_question_data["hint_generated"] = True
                        current_question = st.session_state.current_question_data["question"]
                        llm = Tongyi(model="qwen-turbo", dashscope_api_key=st.secrets["DASHSCOPE_API_KEY"])
                        hint_prompt = f"""请为以下AI产品经理问题生成一个简短的提示（不要直接给出答案），帮助用户思考。提示应该引导思路，比如指出需要考虑的方面或常见分析框架。

题目：{current_question}

只输出提示内容，不要超过100字。"""
                        with st.spinner("生成提示中..."):
                            hint = llm.invoke(hint_prompt).strip()
                        st.session_state.messages.append({"role": "assistant", "content": f"💡 提示：{hint}"})
                        save_current_session()
                        st.rerun()
        with col2:
            prompt = st.chat_input("问问题或输入答案...")
    else:
        prompt = st.chat_input("问问题或输入答案...")

    if prompt:
        # 如果当前处于学习模式（非对话）
        if st.session_state.current_question_data:
            llm = Tongyi(model="qwen-turbo", dashscope_api_key=st.secrets["DASHSCOPE_API_KEY"])
            user_answer = prompt
            st.session_state.messages.append({"role": "user", "content": user_answer})
            with st.chat_message("user"):
                st.markdown(user_answer)
            mode = st.session_state.study_mode
            if mode == "问答题":
                with st.spinner("正在评分中..."):
                    ref_answer = st.session_state.current_question_data["extra"].get("reference_answer")
                    accuracy, error_points, suggestions, detailed = evaluate_answer(
                        user_answer, ref_answer,
                        st.session_state.current_question_data["question"],
                        st.session_state.current_question_data["extra"].get("chunk_metadata")
                    )
                feedback = f"📊 准确率：{accuracy}%\n\n**错误点：** {error_points}"
                if detailed:
                    feedback += f"\n\n{detailed}"
                # 更新掌握度
                chunk_source = st.session_state.current_question_data["extra"].get("chunk_metadata", {}).get("source")
                if chunk_source:
                    mastery = st.session_state.knowledge_mastery
                    if chunk_source not in mastery:
                        mastery[chunk_source] = {"correct": 0, "total": 0}
                    mastery[chunk_source]["total"] += 1
                    if accuracy >= 85:
                        mastery[chunk_source]["correct"] += 1
                
                st.session_state.current_question_data["answered"] = True
                st.session_state.quiz_score += accuracy / 100.0
                st.session_state.quiz_total += 1
                
                with st.chat_message("assistant"):
                    st.markdown(feedback)
                st.session_state.messages.append({"role": "assistant", "content": feedback})
                
                if st.session_state.quiz_total >= 15:
                    end_test_and_save_title("问答题")
                    st.session_state.current_question_data = None
                    with st.chat_message("assistant"):
                        st.markdown("🎉 本次问答题测试结束！")
                    st.session_state.messages.append({"role": "assistant", "content": "🎉 本次问答题测试结束！"})
                    save_current_session()
                    st.rerun()
                else:
                    save_current_session()
                    new_question_text, new_extra_data = generate_study_question(mode)
                    if new_question_text:
                        st.session_state.current_question_data = {
                            "question": new_question_text,
                            "extra": new_extra_data,
                            "question_time": datetime.now(),
                            "answered": False,
                            "hint_generated": False
                        }
                        st.session_state.messages.append({"role": "assistant", "content": new_question_text})
                        save_current_session()
                    else:
                        st.session_state.current_question_data = None
                        st.session_state.messages.append({"role": "assistant", "content": "无法生成下一题，学习结束。"})
                        save_current_session()
                    st.rerun()
            elif mode == "模拟面试":
                process_mock_interview(user_answer)
            elif mode == "场景模拟":
                scenario_text = st.session_state.current_question_data["extra"].get("scenario", "")
                with st.spinner("正在评估场景回答..."):
                    accuracy, error_points, suggestions, detailed = evaluate_scenario(user_answer, scenario_text)
                    feedback = f"📊 准确率：{accuracy}%\n\n**错误点**：{error_points}"
                    if accuracy < 85:
                        if detailed:
                            detailed = detailed.replace('详细建议：', '**详细建议：**')
                            feedback += f"\n\n{detailed}"
                    else:
                        llm_frame = Tongyi(model="qwen-turbo", dashscope_api_key=st.secrets["DASHSCOPE_API_KEY"])
                        frame_prompt = f"""请针对以下AI产品经理场景问题，生成一个5W2H分析表格（What, Why, When, Where, Who, How, How much）。表格格式为Markdown，包含列：维度、分析内容。每个维度给出简要的分析方向，不要给出具体答案。

场景：{scenario_text}

输出格式：
| 维度 | 分析内容 |
|------|----------|
| What | ... |
| Why | ... |
| When | ... |
| Where | ... |
| Who | ... |
| How | ... |
| How much | ... |
"""
                        with st.spinner("生成5W2H分析表格中..."):
                            framework = llm_frame.invoke(frame_prompt).strip()
                        feedback += f"\n\n📐 **5W2H分析表格**\n{framework}"
                    st.session_state.quiz_score += accuracy / 100.0
                    st.session_state.quiz_total += 1
                    with st.chat_message("assistant"):
                        st.markdown(feedback)
                    st.session_state.messages.append({"role": "assistant", "content": feedback})
                    save_current_session()

                    if st.session_state.quiz_total >= 5:
                        end_test_and_save_title("场景模拟")
                        st.session_state.current_question_data = None
                        with st.chat_message("assistant"):
                            st.markdown("🎉 本次场景模拟测试结束！")
                        st.session_state.messages.append({"role": "assistant", "content": "🎉 本次场景模拟测试结束！"})
                        save_current_session()
                        st.rerun()
                    else:
                        new_question_text, new_extra_data = generate_study_question(mode)
                        if new_question_text:
                            st.session_state.current_question_data = {
                                "question": new_question_text,
                                "extra": new_extra_data
                            }
                            st.session_state.messages.append({"role": "assistant", "content": new_question_text})
                            save_current_session()
                        else:
                            st.session_state.current_question_data = None
                            st.session_state.messages.append({"role": "assistant", "content": "无法生成下一题，学习结束。"})
                        st.rerun()
        else:
            # 普通对话模式
            if prompt.strip() in ["出题", "测验", "quiz", "开始测验"]:
                create_new_session(is_quiz=True)
                if st.session_state.vectordb is None:
                    with st.spinner("加载知识库中..."):
                        st.session_state.vectordb = load_knowledge_base()
                question, answer, chunk = generate_quiz_question()
                if question:
                    st.session_state.quiz_mode = True
                    st.session_state.quiz_question = question
                    st.session_state.quiz_answer = answer
                    st.session_state.quiz_chunk = chunk
                    st.session_state.quiz_score = 0.0
                    st.session_state.quiz_total = 0
                    with st.chat_message("assistant"):
                        st.markdown(f"📚 **测验模式已开启**\n\n{question}\n\n请输入你的答案。")
                    st.session_state.messages.append({"role": "assistant", "content": f"📚 **测验模式已开启**\n\n{question}\n\n请输入你的答案。"})
                    save_current_session()
                    st.rerun()
                else:
                    with st.chat_message("assistant"):
                        st.markdown("出题失败，请稍后重试。")
                    st.session_state.messages.append({"role": "assistant", "content": "出题失败，请稍后重试。"})
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                save_current_session()
                if st.session_state.vectordb is None:
                    with st.spinner("加载知识库中..."):
                        st.session_state.vectordb = load_knowledge_base()
                if st.session_state.chain is None:
                    with st.spinner("初始化对话引擎..."):
                        chain, _ = get_conversation_chain(st.session_state.vectordb)
                        st.session_state.chain = chain
                with st.chat_message("assistant"):
                    with st.spinner("思考中..."):
                        result = st.session_state.chain({"question": prompt})
                        answer = result["answer"]
                        source_docs = result.get("source_documents", [])
                        st.markdown(answer)
                        with st.expander("📚 参考来源"):
                            if source_docs:
                                for i, doc in enumerate(source_docs):
                                    st.write(f"**来源 {i+1}:** {doc.metadata.get('source', '未知')}")
                                    st.write(doc.page_content[:200] + "...")
                            else:
                                st.write("无引用来源")
                st.session_state.messages.append({"role": "assistant", "content": answer})
                save_current_session()
                st.rerun()