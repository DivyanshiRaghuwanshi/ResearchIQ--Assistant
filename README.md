# ResearchIQ — Intelligent Research Assistant

A context-aware AI research assistant built for the **NeoStats AI Engineer Case Study**. ResearchIQ combines document-based knowledge retrieval (RAG) with live web search inside a LangGraph ReAct agent, giving users a single interface to query their own files and the open web — with full control over how answers are generated.

**Live Demo:** [Deploy link will go here after Streamlit Cloud deployment]

---

## Why This Exists

The problem with most AI assistants is that they either know only what they were trained on, or they only search the web. Neither is sufficient for research work. You often need to cross-reference your own documents with current information — and you want the AI to decide intelligently which source to use without you having to specify it every time.

ResearchIQ solves this with a ReAct agent that reasons about every query before acting. It picks the right tool, calls it, reads the result, and then composes an answer. The user just asks questions.

---

## Features

### 1. Document Q&A with RAG
Upload your own files and turn them into a searchable knowledge base instantly.

- Supports **PDF, TXT, and DOCX** formats
- Multiple files can be uploaded and merged into one unified index
- Documents are chunked (1000 chars, 200 overlap) and embedded using a local HuggingFace model
- **FAISS KNN search** retrieves the top 7 most semantically relevant chunks per query
- A dedicated retrieval LLM (temperature=0) synthesizes answers strictly from those chunks — no hallucination, no filling in gaps
- Every answer cites its source (document name + page number where available)

### 2. Live Web Search
When your documents don't have the answer — or when you need something current — the agent searches the web.

- Powered by the **Serper API** (Google Search wrapper)
- Returns real-time results with titles, snippets, and source URLs
- The agent decides on its own when a query needs web search vs document search
- Can combine both sources in a single response when the question benefits from it

### 3. Concise and Detailed Response Modes
Not every question needs the same depth of answer.

- **Concise mode:** A focused 2-4 sentence answer. Good for quick lookups or when you already know the topic.
- **Detailed mode:** A structured, multi-section markdown response with headers, explanations, examples, and a summary. Good for understanding something deeply or preparing a writeup.
- Switch between modes from the sidebar at any time mid-conversation.

### 4. Search Mode Toggles
Fine-grained control over what the agent is allowed to use.

- **Document Search (RAG):** Toggle on/off to restrict or allow document queries
- **Web Search:** Toggle on/off to restrict or allow live web queries
- You can run in RAG-only mode (no web), web-only mode (no documents), or hybrid (both)
- The agent automatically rebuilds when you change these settings

### 5. Multi-Provider LLM Support
Switch between AI providers from the sidebar without restarting the app.

- **OpenAI (gpt-4o-mini)** — Default. Recommended for reliable tool calling and consistent responses. Uses a paid API key.
- **Groq (llama-3.1-8b-instant)** — Free tier with generous rate limits. Good for general queries. Some Groq llama models have intermittent tool-call formatting issues; `llama-3.1-8b-instant` is the most stable.
- **Gemini (gemini-1.5-flash)** — Available if you have a properly configured Google Cloud project with billing enabled. Free-tier AI Studio keys may hit quota limits depending on the account.

> **Note on OpenAI as default:** OpenAI's `gpt-4o-mini` is set as the default provider because it produces the most consistent tool-calling behavior and response quality. The project is designed to support all three providers — OpenAI just happens to be the most reliable choice for a live demo with a paid key.

### 6. Persistent Session Memory
The conversation context is maintained throughout your session.

- Built on **LangGraph's MemorySaver (InMemorySaver)** with thread-based checkpointing
- Each browser session gets a unique `thread_id` so context is isolated per user
- Message history is trimmed to the last 20 messages before each LLM call to avoid token limit errors in long conversations
- Clearing the conversation resets memory and generates a new thread

---

## How the Agent Works

The core of ResearchIQ is a **LangGraph ReAct agent** built with `create_react_agent`. Here is what happens from the moment you send a message:

```
User sends a message
        │
        ▼
ReAct Agent receives query + conversation history
        │
        ├── Decides: does this need get_answer, search_web, both, or neither?
        │
        ├── Calls get_answer (RAG tool)
        │       ├── Query embedded with all-MiniLM-L6-v2
        │       ├── FAISS KNN search → top 7 chunks
        │       └── Retrieval LLM (temp=0) → grounded answer with citations
        │
        ├── Calls search_web (Serper tool)
        │       └── Google search results → titles, snippets, URLs
        │
        └── Response LLM (temp=0.3) composes final answer
                │
                ▼
        Answer shown in chat with source citations
```

### Why Two Separate LLMs?

The project uses two LLM instances intentionally:

- **Retrieval LLM at temperature=0** — Used inside the `get_answer` tool when reading document chunks. Zero temperature means the model sticks exactly to what is written in the document. It will not creatively fill in gaps or extrapolate.
- **Response LLM at temperature=0.3** — Used by the main agent to compose the final answer and handle conversation. A small amount of temperature gives responses natural language variation while remaining accurate.

Using a single temperature for both would force a tradeoff between factuality and readability. The dual-LLM design avoids that.

---

## Project Structure

```
project/
│
├── app.py                     # Streamlit UI — sidebar, chat loop, agent orchestration
├── requirements.txt           # All Python dependencies with version pins
├── README.md
│
├── config/
│   └── config.py              # API keys (loaded from secrets), model names, RAG params
│
├── models/
│   ├── llm.py                 # LLM factory — returns Groq/OpenAI/Gemini at given temperature
│   └── embeddings.py          # HuggingFace embeddings setup (normalize_embeddings=True)
│
├── prompts/
│   ├── agent_prompt.py        # Main agent system prompt + concise/detailed mode strings
│   └── rag_prompt.py          # Retrieval LLM prompt (strict: answer only from given chunks)
│
├── utils/
│   ├── rag_utils.py           # Document loading, chunking, FAISS indexing, KNN retrieval
│   ├── search_utils.py        # Serper API call, result formatting, error handling
│   ├── tools.py               # LangChain @tool definitions: get_answer and search_web
│   └── agent_utils.py         # build_agent() and run_agent() using LangGraph
│
└── .streamlit/
    ├── config.toml            # Dark theme, server settings
    └── secrets.toml           # API keys — local only, never committed (gitignored)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent framework | LangGraph `create_react_agent` (langgraph.prebuilt) |
| Session memory | LangGraph `MemorySaver` with thread-based checkpointing |
| LLM — OpenAI | `langchain-openai` → `ChatOpenAI` (gpt-4o-mini) |
| LLM — Groq | `langchain-groq` → `ChatGroq` (llama-3.1-8b-instant) |
| LLM — Gemini | `langchain-google-genai` → `ChatGoogleGenerativeAI` (gemini-1.5-flash) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` — local, runs on CPU, no API cost |
| Vector store | FAISS — in-memory KNN similarity search |
| Text splitting | `langchain-text-splitters` RecursiveCharacterTextSplitter |
| Document loaders | LangChain PDF, DOCX, TXT loaders |
| Web search | Serper API (Google Search) |
| UI | Streamlit |
| Deployment | Streamlit Cloud |

---

## Local Setup

### Prerequisites
- Python 3.9 or higher
- At least one LLM API key (OpenAI recommended)
- Serper API key for web search

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/researchiq-chatbot.git
cd researchiq-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

On first run, the HuggingFace embedding model (`all-MiniLM-L6-v2`) will be downloaded automatically — around 90MB, stored in your local HuggingFace cache.

### 4. Set up API keys

Create the file `.streamlit/secrets.toml` in the project folder (this path is already in `.gitignore` — it will never be committed):

```toml
GROQ_API_KEY    = "your_groq_key"
OPENAI_API_KEY  = "your_openai_key"
GEMINI_API_KEY  = "your_gemini_key"
SERPER_API_KEY  = "your_serper_key"
```

You only need the keys for the providers you plan to use. OpenAI + Serper is the recommended minimum.

| Service | Where to get it | Cost |
|---|---|---|
| OpenAI | [platform.openai.com](https://platform.openai.com) | Paid (pay-per-token) |
| Groq | [console.groq.com](https://console.groq.com) | Free tier available |
| Gemini | Google Cloud project with billing enabled | Free tier limited |
| Serper | [serper.dev](https://serper.dev) | 2,500 free searches/month |

### 5. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Using the App

1. **Select LLM Provider** from the sidebar (OpenAI is default and recommended).
2. **Choose Response Mode** — Concise for quick answers, Detailed for structured explanations.
3. **Search Mode** — Both RAG and web search are enabled by default. Uncheck either to restrict the agent to one source.
4. **Upload Documents** — Drag your PDFs, Word files, or text files into the upload box. You'll see a chunk count confirming the index was built.
5. **Start chatting** — Ask anything. The agent will pick the right tool automatically.

### Example Queries

**With documents uploaded:**
- "Summarize this document"
- "What projects are mentioned in the resume?"
- "Explain the Self-Supervised Methodology section in detail"
- "What are the key findings in chapter 3?"

**Web search:**
- "Latest developments in LLM agent frameworks"
- "What is the current state of RAG in production?"
- "Recent AI research on vector databases"

**Hybrid (both tools):**
- "What does the document say about transformers, and what are the latest improvements to that architecture?"

---

## Configuration Reference

All tunable parameters live in `config/config.py`:

| Parameter | Default | What it controls |
|---|---|---|
| `DEFAULT_LLM_PROVIDER` | `"openai"` | Which provider loads on startup |
| `RETRIEVAL_TEMPERATURE` | `0.0` | LLM temperature for RAG extraction (factual) |
| `RESPONSE_TEMPERATURE` | `0.3` | LLM temperature for final agent response |
| `CHUNK_SIZE` | `1000` | Characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks for context continuity |
| `TOP_K` | `7` | Number of chunks retrieved per query |
| `MAX_HISTORY_MESSAGES` | `20` | Messages kept in context window per session |
| `SERPER_NUM_RESULTS` | `5` | Web search results returned per query |

---

## Deployment on Streamlit Cloud

### Step 1 — Push to GitHub

```bash
git add .
git commit -m "initial commit"
git push origin main
```

### Step 2 — Deploy

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Connect your GitHub account and select the repository
4. Set **Main file path** to `app.py`
5. Click **Advanced settings** and open the **Secrets** tab
6. Paste your API keys in TOML format:

```toml
GROQ_API_KEY    = "your_groq_key"
OPENAI_API_KEY  = "your_openai_key"
GEMINI_API_KEY  = "your_gemini_key"
SERPER_API_KEY  = "your_serper_key"
```

7. Click **Deploy**

Streamlit Cloud will install dependencies from `requirements.txt` and start the app. The HuggingFace model downloads automatically on first boot.

---

## Design Decisions

**Why LangGraph instead of a basic LangChain agent?**
LangGraph's `create_react_agent` with `MemorySaver` gives proper checkpoint-based memory tied to a `thread_id`. This means each user session is truly isolated — even in a shared deployment, two users won't see each other's conversation history. A basic LangChain agent with a message list doesn't give you this.

**Why local embeddings instead of OpenAI embeddings?**
`all-MiniLM-L6-v2` runs entirely on CPU with no API calls. This means document indexing is free regardless of file size, and there's no network dependency during the embedding step. For a demo and case study submission, this is the right call. A production deployment targeting scale would switch to a hosted embedding service.

**Why FAISS over a cloud vector database?**
FAISS is in-memory and has zero infrastructure overhead. For a single-session demo, it's perfectly appropriate. The architecture is modular enough that swapping FAISS for Pinecone or Weaviate would only require changing `rag_utils.py` — the rest of the system doesn't care.

**Why two separate LLM instances at different temperatures?**
Covered in the architecture section above. Short version: temperature=0 for fact extraction from documents, temperature=0.3 for conversational responses. Mixing these would compromise one or the other.

---

## Security Notes

- API keys are loaded from `.streamlit/secrets.toml` locally and from Streamlit Cloud Secrets in production
- `secrets.toml` is in `.gitignore` and will never appear in the repository
- No API keys are displayed anywhere in the UI
- User-uploaded documents are processed in memory and are not persisted to disk between sessions

---

## Troubleshooting

**App won't start / import errors**
Make sure you're running from inside the `project/` folder and your virtual environment is activated.
```bash
cd project
streamlit run app.py
```

**"No module named X" error**
```bash
pip install -r requirements.txt
```

**Groq XML tool-call error** (`tool_use_failed`)
This is a known issue with certain Groq llama models generating `<function=...>` format instead of JSON. Switch to OpenAI from the sidebar or use `llama-3.1-8b-instant` (most stable Groq option).

**Gemini quota errors**
Free-tier Gemini API keys from Google AI Studio may have zero quota on newer models. This requires a Google Cloud project with billing enabled. OpenAI or Groq are the recommended alternatives.

**Document upload shows 0 chunks**
The uploaded PDF may be image-based (scanned) rather than text-based. Text-based PDFs have extractable content. Image-based PDFs require OCR which is not included in this version.

**Port already in use**
Another Streamlit instance is running. Kill it first:
```bash
# Windows PowerShell
Get-Process python | Stop-Process -Force
streamlit run app.py
```
