"""
app.py — Adaptive Multi-Agent Tech Support Network
Streamlit front-end for the Orion SmartHub X1 Pro Support System

This app uses the EXACT same agent logic as the working notebook.ipynb.

Run locally:
    streamlit run app.py

Deploy on Streamlit Cloud:
    Push this file + requirements.txt to GitHub,
    then connect the repo at https://share.streamlit.io
    Set OPENAI_API_KEY in App Settings -> Secrets.
"""

import os
import streamlit as st
from typing import List
from typing_extensions import TypedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.documents import Document
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END


# ══════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(page_title="Orion Support Hub", page_icon="🤖", layout="centered")
st.title("🤖 Orion SmartHub X1 Pro — AI Support Hub")
st.caption("Powered by a Multi-Agent LangGraph Network | ChromaDB · DuckDuckGo · GPT-4o-mini")
st.divider()


# ══════════════════════════════════════════════════════════
# API KEY — from Streamlit Secrets or sidebar input
# ══════════════════════════════════════════════════════════
api_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""

if not api_key:
    with st.sidebar:
        st.header("🔑 Configuration")
        api_key = st.text_input(
            "OpenAI API Key", type="password", placeholder="sk-...",
            help="Your key is only used for this session and never stored."
        )
        st.caption("Get a key at [platform.openai.com](https://platform.openai.com)")

if not api_key:
    st.warning("⬅️ Enter your OpenAI API Key in the sidebar to start.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key


# ══════════════════════════════════════════════════════════
# GRAPH STATE  (identical to notebook)
# ══════════════════════════════════════════════════════════
class GraphState(TypedDict):
    original_query:   str
    optimized_query:  str
    documents:        List[Document]
    generation:       str
    loop_count:       int
    web_search_used:  bool
    qa_trace:         List[str]


# ══════════════════════════════════════════════════════════
# FAQ SOURCE TEXT  (the Orion manual)
# ══════════════════════════════════════════════════════════
FAQ_TEXT = """
ENTERPRISE TECHNICAL MANUAL & FAQ: ORION SMARTHUB X1 PRO GATEWAY
Product Engine: Orion Core Ecosystem
Hardware Model: SmartHub-X1-PRO
Firmware Baseline: Stable Build v4.12.0

1. HARDWARE SPECIFICATIONS & INTERNAL CHIPSETS
The Orion SmartHub X1 Pro serves as a high-performance local bridge
and supervisor node for hybrid smart home fabrics.
- Main Processor: Quad-core 1.8GHz ARM Cortex-A53 Compute Unit
- Neural Engine: Integrated 0.5 TOPS Edge-AI core for offline voice intent processing
- System Memory: 2GB DDR4 Non-ECC RAM
- Storage Architecture: 16GB eMMC 5.1 Flash partitions
- Network Stack: Dual-Band Wi-Fi (2.4GHz / 5GHz 802.11ax), Gigabit Ethernet RJ-45, Bluetooth Low Energy (BLE v5.2)
- Power Delivery: Dedicated 12V/1.5A DC input negotiated via USB-C PD 3.0

2. PROVISIONING AND SYSTEM INITIALIZATION
To register and commission a new SmartHub X1 Pro on an enterprise or residential network:
Step 1: Inspect the hardware for integrity, connect the OEM USB-C power supply, and connect to a power outlet.
Step 2: Monitor the chassis faceplate. Wait for the primary LED to blink in an Amber color pattern, signaling Ready for Provisioning.
Step 3: Launch the Orion Home Mobile Application (v6.4 or higher) or access the provisioning gateway at http://192.168.1.1 from a local subnet.
Step 4: Click the + action button in the upper-right corner of the app console and select Discover New Gateway Hub.
Step 5: Position your mobile device camera to scan the secure cryptographic QR code stamped onto the bottom silicone footing of the SmartHub unit.
Step 6: Input your local network Wi-Fi SSID and passphrase when prompted.
Step 7: Allow the unit up to 90 seconds to finalize encryption handshakes with cloud servers. Once verified, the LED status light will transition to a Solid Sapphire Blue state.

3. HARDWARE LED DIAGNOSTIC LIGHT MATRIX
The SmartHub X1 Pro uses an embedded RGB diode assembly to relay real-time system health metrics.
- Solid Sapphire Blue: Hardware operating normally. Core system partitions are active, local loops are functional, and cloud server WAN socket connections are live.
- Blinking Amber: Provisioning mode active. The system is unauthenticated and broadcasting its pairing beacon.
- Flashing Crimson Red: Critical kernel error detected. This pattern uniquely points to a system boot partition validation failure or a corrupted local data cache block.
- Solid Emerald Green: WAN Disconnect / Local Fallback Active. The local home automation engine remains fully operational, but active WAN internet connectivity has been completely dropped.

4. SYSTEM RESTORATION & HARD RESET ARCHITECTURE
Soft System Reboot: Log into the cloud dashboard panel, navigate to System Admin Console > Maintenance Tools, and commit the Restart Hub pipeline. Hardware method: Disconnect the power coupling from the USB-C terminal port for exactly 15 seconds, then re-insert.
Hard Factory Reset: Locate the recessed sub-surface microswitch button labeled RESET on the rear panel assembly (situated directly to the left of the physical RJ-45 Ethernet port). Using a steel paperclip or SIM ejector pin, depress and hold this internal switch continuously for exactly 12 seconds. The LED assembly will blink Crimson Red rapidly three consecutive times, confirming that all persistent storage pools, cryptographic pairings, and custom automation scripts have been wiped clean.

5. ERROR RESOLUTION: PROTOCOL CODE ERR-302 (Z-WAVE MESH DROPOUT)
The system throws an ERR-302 trace on the administrative dashboard when the Z-Wave controller co-processor registers excessive localized electromagnetic noise floor thresholds.
Remediation Protocol:
1. Physical Isolation: Ensure the SmartHub enclosure is positioned a minimum of 3 feet (1 meter) away from microwave ovens, large consumer appliances, or high-power Wi-Fi access points.
2. Mesh Topology Recovery: Open the Developer Console, expand the Z-Wave Stack Management subsection, and invoke the Network Heal directive. This prompts the network controller to run neighbor rediscovery passes and rewrite optimized routing paths.
3. Cold Cycle Fallback: If the alert trace does not auto-clear within 10 continuous minutes, initiate a standard Soft System Reboot.
"""


# ══════════════════════════════════════════════════════════
# BUILD VECTOR STORE + GRAPH  (cached — built once per session)
# Logic is identical to the working notebook.
# ══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="📚 Building vector database from FAQ manual...")
def build_app():

    # ── Split + embed into ChromaDB
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    raw_chunks = splitter.split_text(FAQ_TEXT)
    docs = [Document(page_content=c, metadata={"source": "orion_hub_manual"})
            for c in raw_chunks]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=docs, embedding=embeddings, collection_name="orion_faq"
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # ══════════════════════════════════════════════
    # NODE 1 — Query Rewriter (Support Agent)
    # ══════════════════════════════════════════════
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a technical support query optimizer. "
         "Your ONLY job is to rewrite the user's question into a concise, "
         "keyword-rich search query optimized for semantic vector search. "
         "Output ONLY the rewritten query — no explanation, no preamble."),
        ("human", "Original question: {query}")
    ])
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    def node_rewrite_query(state: GraphState) -> GraphState:
        optimized = rewrite_chain.invoke({"query": state["original_query"]})
        trace = state.get("qa_trace", [])
        trace.append(f"✏️ Query rewritten: '{state['original_query']}' → '{optimized}'")
        return {**state, "optimized_query": optimized, "qa_trace": trace}

    # ══════════════════════════════════════════════
    # NODE 2 — Retrieve & Grade (Support Agent + QA Agent)
    # ══════════════════════════════════════════════
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a QA Engineer grading document relevance. "
         "Given the user question and a retrieved document chunk, "
         "decide if the chunk is relevant to answering the question. "
         'Output ONLY valid JSON: {{"relevance_score": "yes"}} or {{"relevance_score": "no"}}'),
        ("human", "Question: {question}\n\nDocument: {document}")
    ])
    grade_chain = grade_prompt | llm | JsonOutputParser()

    def node_retrieve_and_grade(state: GraphState) -> GraphState:
        raw_docs = retriever.invoke(state["optimized_query"])
        relevant_docs = []
        for doc in raw_docs:
            result = grade_chain.invoke({
                "question": state["optimized_query"],
                "document": doc.page_content
            })
            if result.get("relevance_score") == "yes":
                relevant_docs.append(doc)
        trace = state.get("qa_trace", [])
        trace.append(
            f"📚 Retrieved {len(raw_docs)} chunks → {len(relevant_docs)} graded relevant by QA Agent"
        )
        return {**state, "documents": relevant_docs, "qa_trace": trace}

    # ══════════════════════════════════════════════
    # NODE 3 — Web Search Fallback (DuckDuckGo)
    # ══════════════════════════════════════════════
    web_search_tool = DuckDuckGoSearchRun()

    def node_web_search(state: GraphState) -> GraphState:
        results = web_search_tool.invoke(state["optimized_query"])
        web_doc = Document(
            page_content=results,
            metadata={"source": "web_search", "query": state["optimized_query"]}
        )
        trace = state.get("qa_trace", [])
        trace.append("🌐 Local docs insufficient — Web Search triggered (DuckDuckGo)")
        return {**state, "documents": [web_doc], "web_search_used": True, "qa_trace": trace}

    # ══════════════════════════════════════════════
    # NODE 4 — Generate Answer (Support Agent)
    # ══════════════════════════════════════════════
    generate_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior technical support specialist. "
         "Using ONLY the provided context documents, write a clear, accurate, "
         "and complete technical answer to the user's question. "
         "Do NOT invent facts. If the context does not contain the answer, say so clearly. "
         "Format your answer with numbered steps when describing procedures."),
        ("human",
         "User Question: {question}\n\n"
         "Context Documents:\n{context}\n\n"
         "Write a professional technical support answer:")
    ])
    generate_chain = generate_prompt | llm | StrOutputParser()

    def node_generate(state: GraphState) -> GraphState:
        context = "\n\n---\n\n".join([d.page_content for d in state["documents"]])
        answer = generate_chain.invoke({
            "question": state["original_query"],
            "context": context
        })
        trace = state.get("qa_trace", [])
        trace.append("💬 Answer generated by Support Agent")
        return {**state, "generation": answer, "qa_trace": trace}

    # ══════════════════════════════════════════════
    # NODE 5 — Hallucination Check (QA Agent / Gatekeeper)
    # ══════════════════════════════════════════════
    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict QA Engineer performing a hallucination audit. "
         "Check whether the generated answer is FULLY grounded in the provided source documents. "
         "If every factual claim in the answer can be traced to the source documents, output PASSED. "
         "If the answer contains any invented facts, unsupported claims, or speculation, output FAILED. "
         'Output ONLY valid JSON: {{"hallucination_check": "passed"}} or {{"hallucination_check": "failed"}}'),
        ("human",
         "Source Documents:\n{documents}\n\n"
         "Generated Answer:\n{generation}")
    ])
    hallucination_chain = hallucination_prompt | llm | JsonOutputParser()

    def node_hallucination_check(state: GraphState) -> GraphState:
        docs_text = "\n\n---\n\n".join([d.page_content for d in state["documents"]])
        result = hallucination_chain.invoke({
            "documents": docs_text,
            "generation": state["generation"]
        })
        verdict = result.get("hallucination_check", "failed")
        trace = state.get("qa_trace", [])
        if verdict == "passed":
            trace.append("✅ QA Agent: Hallucination check PASSED — answer is grounded")
            return {**state, "qa_trace": trace}
        else:
            new_count = state.get("loop_count", 0) + 1
            trace.append(
                f"❌ QA Agent: Hallucination check FAILED (loop {new_count}/2) — triggering rewrite"
            )
            return {**state, "loop_count": new_count, "qa_trace": trace}

    # ══════════════════════════════════════════════
    # EDGE ROUTERS
    # ══════════════════════════════════════════════
    def route_after_grading(state: GraphState) -> str:
        if state["documents"]:
            return "generate"
        else:
            return "web_search"

    def route_after_hallucination_check(state: GraphState) -> str:
        trace = state.get("qa_trace", [])
        last_entry = trace[-1] if trace else ""
        if "PASSED" in last_entry:
            return "end"
        loop_count = state.get("loop_count", 0)
        if loop_count >= 2:
            return "end"
        return "rewrite"

    # ══════════════════════════════════════════════
    # COMPILE GRAPH
    # ══════════════════════════════════════════════
    gb = StateGraph(GraphState)
    gb.add_node("rewrite_query",       node_rewrite_query)
    gb.add_node("retrieve_and_grade",  node_retrieve_and_grade)
    gb.add_node("web_search",          node_web_search)
    gb.add_node("generate",            node_generate)
    gb.add_node("hallucination_check", node_hallucination_check)

    gb.set_entry_point("rewrite_query")
    gb.add_edge("rewrite_query",  "retrieve_and_grade")
    gb.add_edge("web_search",     "generate")
    gb.add_edge("generate",       "hallucination_check")

    gb.add_conditional_edges(
        "retrieve_and_grade", route_after_grading,
        {"generate": "generate", "web_search": "web_search"}
    )
    gb.add_conditional_edges(
        "hallucination_check", route_after_hallucination_check,
        {"end": END, "rewrite": "rewrite_query"}
    )

    return gb.compile()


# ── Build the graph (cached)
support_app = build_app()
st.success("✅ AI Support Network is ready!", icon="🚀")


# ══════════════════════════════════════════════════════════
# CHAT HISTORY
# ══════════════════════════════════════════════════════════
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "trace" in msg:
            with st.expander("🔍 View Agentic Decision Trace", expanded=False):
                for step in msg["trace"]:
                    st.markdown(f"- {step}")


# ══════════════════════════════════════════════════════════
# CHAT INPUT
# ══════════════════════════════════════════════════════════
if user_input := st.chat_input("Ask a technical support question about your Orion SmartHub..."):

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.status("🤖 Multi-agent network processing your query...", expanded=True) as status:
            st.write("✏️ Support Agent rewriting query...")

            initial_state: GraphState = {
                "original_query":  user_input,
                "optimized_query": "",
                "documents":       [],
                "generation":      "",
                "loop_count":      0,
                "web_search_used": False,
                "qa_trace":        []
            }

            final_state = support_app.invoke(initial_state)

            trace    = final_state.get("qa_trace", [])
            web_used = final_state.get("web_search_used", False)
            loops    = final_state.get("loop_count", 0)

            if loops > 0:
                status.update(label=f"✅ Done — QA correction loop triggered {loops}x", state="complete")
            elif web_used:
                status.update(label="✅ Done — Web search fallback used", state="complete")
            else:
                status.update(label="✅ Done — Answered from local knowledge base", state="complete")

        answer = final_state.get("generation", "⚠️ No answer could be generated.")
        st.markdown(answer)

        with st.expander("🔍 View Agentic Decision Trace", expanded=False):
            for step in trace:
                st.markdown(f"- {step}")

    st.session_state.chat_history.append({
        "role": "assistant", "content": answer, "trace": trace
    })


# ══════════════════════════════════════════════════════════
# SIDEBAR — Info & Sample Questions
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.divider()
    st.subheader("💡 Try These Queries")
    for q in [
        "My LED is solid green. Is the device working?",
        "How do I do a factory reset?",
        "What does ERR-302 mean and how do I fix it?",
        "How do I set up the SmartHub for the first time?",
        "What processor does the SmartHub use?",
    ]:
        st.code(q, language=None)

    st.divider()
    st.subheader("🏗️ Agent Architecture")
    st.markdown("""
**Support Specialist Agent**
- Rewrites & optimizes queries
- Retrieves from ChromaDB (top 3 chunks)
- Generates grounded answers

**QA Gatekeeper Agent**
- Grades chunk relevance (`yes/no`)
- Checks for hallucinations (`passed/failed`)
- Triggers self-correction loops (max 2)

**Web Fallback**
- DuckDuckGo (free, no API key)
- Only fires when local docs are insufficient
""")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
