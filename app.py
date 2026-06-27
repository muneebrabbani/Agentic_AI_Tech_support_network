"""
app.py — Adaptive Multi-Agent Tech Support Network
Streamlit front-end for the Orion SmartHub X1 Pro Support System

Run locally:
    streamlit run app.py

Deploy on Streamlit Cloud:
    Push this file + requirements.txt + orion_hub_manual.txt to GitHub,
    then connect the repo at https://share.streamlit.io
    Set OPENAI_API_KEY in App Settings → Secrets.
"""

import os
import streamlit as st
from typing import List
from typing_extensions import TypedDict

# ── Lazy imports so Streamlit can boot even before packages are installed
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
st.set_page_config(
    page_title="Orion Support Hub",
    page_icon="🤖",
    layout="centered",
)

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
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Your key is only used for this session and never stored."
        )
        st.caption("Get a key at [platform.openai.com](https://platform.openai.com)")

if not api_key:
    st.warning("⬅️ Enter your OpenAI API Key in the sidebar to start.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key


# ══════════════════════════════════════════════════════════
# GRAPH STATE
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
# VECTOR STORE — built once and cached across sessions
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


@st.cache_resource(show_spinner="📚 Building vector database from FAQ manual...")
def build_vectorstore_and_graph():
    """Build ChromaDB vector store and compile LangGraph — cached for the session."""

    # ── 1. Chunk the FAQ document
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(FAQ_TEXT)
    docs = [
        Document(page_content=c, metadata={"source": "orion_hub_manual"})
        for c in chunks
    ]

    # ── 2. Embed and store in ChromaDB (in-memory for Streamlit Cloud)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="orion_faq"
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # ── 3. LLM (single instance)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # ── 4. Chains
    rewrite_chain = (
        ChatPromptTemplate.from_messages([
            ("system",
             "You are a technical support query optimizer for the Orion SmartHub X1 Pro gateway device. "
             "Rewrite the user's question into a concise keyword-rich search query. "
             "IMPORTANT: Always keep Orion SmartHub X1 Pro product context in the query. "
             "Include hardware terms like: LED color, reset, provisioning, ERR-302, Z-Wave, "
             "firmware, setup, Wi-Fi, Ethernet, BLE whenever relevant. "
             "Output ONLY the rewritten query, nothing else."),
            ("human", "Original question: {query}")
        ]) | llm | StrOutputParser()
    )

    grade_chain = (
        ChatPromptTemplate.from_messages([
            ("system",
             "You are a QA Engineer grading document relevance for the Orion SmartHub X1 Pro manual. "
             "Grade YES if the document chunk contains ANY information that could help answer the question — "
             "even partial matches, related hardware concepts, or adjacent topics count as relevant. "
             "Only grade NO if the chunk is completely unrelated to the question topic. "
             "Be generous: when in doubt, grade yes. "
             'Output ONLY JSON: {{"relevance_score": "yes"}} or {{"relevance_score": "no"}}'),
            ("human", "Question: {question}\n\nDocument: {document}")
        ]) | llm | JsonOutputParser()
    )

    generate_chain = (
        ChatPromptTemplate.from_messages([
            ("system",
             "You are a senior technical support specialist for the Orion SmartHub X1 Pro gateway. "
             "Answer the user's question using STRICTLY and ONLY the context documents provided below. "
             "Rules you must follow:\n"
             "1. Do NOT use any outside knowledge. Every sentence must be traceable to the context.\n"
             "2. If the context does not contain the answer, say exactly: "
             "'I could not find this information in the Orion SmartHub X1 Pro manual.'\n"
             "3. Never guess, infer, or add information beyond what is explicitly stated in the context.\n"
             "4. Use numbered steps when describing procedures.\n"
             "5. Quote or closely paraphrase the manual — do not generalize."),
            ("human",
             "User Question: {question}\n\n"
             "Context Documents (use ONLY these):\n{context}\n\n"
             "Answer strictly based on the context above:")
        ]) | llm | StrOutputParser()
    )

    hallucination_chain = (
        ChatPromptTemplate.from_messages([
            ("system",
             "You are a QA Engineer performing a hallucination audit. "
             "Is every factual claim in the answer grounded in the source documents? "
             'Output ONLY JSON: {{"hallucination_check": "passed"}} or {{"hallucination_check": "failed"}}'),
            ("human",
             "Source Documents:\n{documents}\n\nGenerated Answer:\n{generation}")
        ]) | llm | JsonOutputParser()
    )

    web_tool = DuckDuckGoSearchRun()

    # ── 5. Node functions
    def node_rewrite_query(state: GraphState) -> GraphState:
        optimized = rewrite_chain.invoke({"query": state["original_query"]})
        trace = state.get("qa_trace", [])
        trace.append(f"✏️ Query rewritten: **'{state['original_query']}'** → **'{optimized}'**")
        return {**state, "optimized_query": optimized, "qa_trace": trace}

    def node_retrieve_and_grade(state: GraphState) -> GraphState:
        raw_docs = retriever.invoke(state["optimized_query"])
        relevant = []
        for doc in raw_docs:
            r = grade_chain.invoke({
                "question": state["optimized_query"],
                "document": doc.page_content
            })
            if r.get("relevance_score") == "yes":
                relevant.append(doc)
        trace = state.get("qa_trace", [])
        trace.append(
            f"📚 Retrieved {len(raw_docs)} chunks → **{len(relevant)} graded relevant** by QA Agent"
        )
        return {**state, "documents": relevant, "qa_trace": trace}

    def node_web_search(state: GraphState) -> GraphState:
        # Anchor web search to product to avoid generic off-topic results
        #anchored_query = f"Orion SmartHub X1 Pro {state['optimized_query']}"
        anchored_query = f" {state['optimized_query']}"
        results = web_tool.invoke(anchored_query)
        web_doc = Document(
            page_content=results,
            metadata={"source": "web_search"}
        )
        trace = state.get("qa_trace", [])
        trace.append("🌐 Local docs insufficient — **Web Search triggered** (DuckDuckGo fallback)")
        return {**state, "documents": [web_doc], "web_search_used": True, "qa_trace": trace}

    def node_generate(state: GraphState) -> GraphState:
        context = "\n\n---\n\n".join([d.page_content for d in state["documents"]])
        answer = generate_chain.invoke({
            "question": state["original_query"],
            "context": context
        })
        trace = state.get("qa_trace", [])
        trace.append("💬 Answer drafted by **Support Specialist Agent**")
        return {**state, "generation": answer, "qa_trace": trace}

    def node_hallucination_check(state: GraphState) -> GraphState:
        docs_text = "\n\n---\n\n".join([d.page_content for d in state["documents"]])
        result = hallucination_chain.invoke({
            "documents": docs_text,
            "generation": state["generation"]
        })
        verdict = result.get("hallucination_check", "failed")
        trace = state.get("qa_trace", [])
        if verdict == "passed":
            trace.append("✅ **QA Gatekeeper: PASSED** — answer is fully grounded in sources")
            return {**state, "qa_trace": trace}
        else:
            new_count = state.get("loop_count", 0) + 1
            trace.append(
                f"❌ **QA Gatekeeper: FAILED** (attempt {new_count}/2) — triggering self-correction loop"
            )
            return {**state, "loop_count": new_count, "qa_trace": trace}

    # ── 6. Edge routers
    def route_after_grading(state: GraphState) -> str:
        return "generate" if state["documents"] else "web_search"

    def route_after_hallucination(state: GraphState) -> str:
        trace = state.get("qa_trace", [])
        last = trace[-1] if trace else ""
        if "PASSED" in last:
            return "end"
        return "end" if state.get("loop_count", 0) >= 2 else "rewrite"

    # ── 7. Build and compile graph
    gb = StateGraph(GraphState)
    gb.add_node("rewrite_query",       node_rewrite_query)
    gb.add_node("retrieve_and_grade",  node_retrieve_and_grade)
    gb.add_node("web_search",          node_web_search)
    gb.add_node("generate",            node_generate)
    gb.add_node("hallucination_check", node_hallucination_check)

    gb.set_entry_point("rewrite_query")
    gb.add_edge("rewrite_query",     "retrieve_and_grade")
    gb.add_edge("web_search",        "generate")
    gb.add_edge("generate",          "hallucination_check")
    gb.add_conditional_edges(
        "retrieve_and_grade",
        route_after_grading,
        {"generate": "generate", "web_search": "web_search"}
    )
    gb.add_conditional_edges(
        "hallucination_check",
        route_after_hallucination,
        {"end": END, "rewrite": "rewrite_query"}
    )

    compiled_app = gb.compile()
    return compiled_app


# ── Build/load the graph (cached)
support_app = build_vectorstore_and_graph()
st.success("✅ AI Support Network is ready!", icon="🚀")


# ══════════════════════════════════════════════════════════
# CHAT HISTORY
# ══════════════════════════════════════════════════════════
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Render previous messages
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

    # Show user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run the multi-agent graph
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

            trace = final_state.get("qa_trace", [])
            web_used = final_state.get("web_search_used", False)
            loops = final_state.get("loop_count", 0)

            # Update status label
            if loops > 0:
                status.update(
                    label=f"✅ Done — QA correction loop triggered {loops}x",
                    state="complete"
                )
            elif web_used:
                status.update(label="✅ Done — Web search fallback used", state="complete")
            else:
                status.update(label="✅ Done — Answered from local knowledge base", state="complete")

        # Display the answer
        answer = final_state.get("generation", "⚠️ No answer could be generated.")
        st.markdown(answer)

        # Show the agentic trace in an expander
        with st.expander("🔍 View Agentic Decision Trace", expanded=False):
            for step in trace:
                st.markdown(f"- {step}")

    # Save to chat history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "trace": trace
    })


# ══════════════════════════════════════════════════════════
# SIDEBAR — Info & Sample Questions
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.divider()
    st.subheader("💡 Try These Queries")
    sample_qs = [
        "My LED is solid green. Is the device working?",
        "How do I do a factory reset?",
        "What does ERR-302 mean and how do I fix it?",
        "How do I set up the SmartHub for the first time?",
        "What processor does the SmartHub use?",
    ]
    for q in sample_qs:
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