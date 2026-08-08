# Advanced LLM & Agentic AI Experiments Report

This document provides a comprehensive analysis, architectural breakdown, and comparison of the LLM and Agentic AI experiments conducted in the workspace. These experiments range from basic single-prompt setups to advanced, production-oriented pipelines integrating vector databases (FAISS), sentence embeddings, relational databases (SQLite), and agentic multi-step planning.

---

## 1. Overview of the Experiments

The experiments are divided into two main environments:
1. **Basic/Educational Implementations** (`agentic ai/llm_assignments/`): Focused on demonstrating core concepts (Prompting, Chaining, Planning, basic RAG).
2. **Production-Ready/Advanced Workflows** (`advanced_workflows/`): Incorporating validation checks, semantic indexing of database schemas, strict grounding, and error-handling mechanisms.

### Experiment Catalog

| Experiment | Target File | Core Objective | Primary Technologies | Complexity |
| :--- | :--- | :--- | :--- | :--- |
| **01. Simple LLM Call** | [q1_llm_workflow.py](file:///c:/Users/D.Megha%20vardhan/OneDrive/Desktop/advanced_workflows/q1_llm_workflow.py) | Establish basic connection and zero-shot question answering. | `google-genai` (Gemini 3.5 Flash) | Basic |
| **02. Prompt Chaining (Basic)** | [q2_prompt_chaining.py](file:///c:/Users/D.Megha%20vardhan/OneDrive/Desktop/agentic%20ai/llm_assignments/q2_prompt_chaining.py) | Sequentially generate a summary, key points, and related questions. | `google-genai` | Basic |
| **03. Prompt Chaining (Prod)** | [prompt_chaining.py](file:///c:/Users/D.Megha%20vardhan/OneDrive/Desktop/advanced_workflows/prompt_chaining.py) | Summarize, extract insights, and synthesize an executive brief with length validations. | `google-genai` | Medium |
| **04. Agentic AI (Planning)** | [q3_agentic_ai.py](file:///c:/Users/D.Megha%20vardhan/OneDrive/Desktop/agentic%20ai/llm_assignments/q3_agentic_ai.py) | Implement a Plan-then-Execute pattern where the model designs and executes a checklist. | `google-genai` | Medium |
| **05. RAG QA (Basic)** | [q4_rag_qa.py](file:///c:/Users/D.Megha%20vardhan/OneDrive/Desktop/agentic%20ai/llm_assignments/q4_rag_qa.py) | Basic QA retrieval using paragraph chunks and 1-NN FAISS search on static data. | `sentence-transformers`, `faiss`, `google-genai` | Medium |
| **06. RAG QA (Prod)** | [rag_qa.py](file:///c:/Users/D.Megha%20vardhan/OneDrive/Desktop/advanced_workflows/rag_qa.py) | Context-grounded Q&A on detailed product documentation using 2-NN retrieval and fallback guards. | `sentence-transformers`, `faiss`, `google-genai` | High |
| **07. Text-to-SQL Engine** | [text_to_sql.py](file:///c:/Users/D.Megha%20vardhan/OneDrive/Desktop/advanced_workflows/text_to_sql.py) | Semantic schema retrieval, SQLite generation/execution, and natural language response synthesis. | `sentence-transformers`, `faiss`, `sqlite3`, `google-genai` | High |

---

## 2. Deep Dive: Architectural Workflows

Below are the detailed workflows for the key architectural patterns tested in these experiments.

### A. Prompt Chaining Pipeline (Production-Ready)
Prompt chaining decomposes a complex task into multiple, smaller prompts where the output of one step becomes the input to the next. The production script includes validation checks to prevent failures from propagating downstream.

```mermaid
graph TD
    A[Input Text] --> B{Length > 50 chars?}
    B -- No --> C[Print Error & Stop]
    B -- Yes --> D[Step 1: Detailed Summary]
    D --> E{Summary Valid?}
    E -- No --> F[Stop Pipeline]
    E -- Yes --> G[Step 2: Extract Key Insights]
    G --> H{Insights Valid?}
    H -- No --> I[Stop Pipeline]
    H -- Yes --> J[Step 3: Executive Brief < 120 words]
    J --> K[Final Output Response]
```

### B. Retrieval-Augmented Generation (RAG) System
RAG overcomes LLM context limitations and knowledge cutoff dates by dynamically retrieving relevant document parts from a vector space before querying the LLM.

```mermaid
flowchart TD
    subgraph Ingestion Phase
        Doc[sample_document.txt] --> Chunk[Chunk by Paragraph / Double Newline]
        Chunk --> Embed[Embed chunks with all-MiniLM-L6-v2]
        Embed --> FAISS[(FAISS IndexFlatL2)]
    end

    subgraph Query Phase
        Query[User Query] --> QEmbed[Embed Query]
        QEmbed --> Search[Search FAISS for k=2 nearest neighbors]
        FAISS --> Search
        Search --> Prompt[Format QA Prompt with Retrieved Context]
        Prompt --> LLM[Gemini 3.5 Flash]
        LLM --> Answer[Grounded Natural Language Answer]
    end
```

### C. Text-to-SQL Query Engine with Semantic Schema Selection
Instead of passing the entire database schema to the LLM context (which is expensive and error-prone), this engine retrieves only the relevant tables using vector similarity on table descriptions.

```mermaid
flowchart TD
    subgraph Schema Setup
        SchemaDef[Table Schemas + Descriptions] --> EmbedSchema[Embed Table Descriptions]
        EmbedSchema --> FAISS_SQL[(FAISS Schema Index)]
    end

    subgraph Query Execution
        UserQ[User Question] --> EmbedQ[Embed User Question]
        EmbedQ --> RetrieveTables[Retrieve Top k=2 Table DDLs]
        FAISS_SQL --> RetrieveTables
        RetrieveTables --> PromptSQL[Prompt Gemini to write SQLite query]
        PromptSQL --> GenSQL[Gemini generates raw SQL]
        GenSQL --> ExecuteSQL[Execute on SQLite Database]
        ExecuteSQL --> CheckError{Success?}
        CheckError -- Yes --> Synthesize[Synthesize NL Answer from DB Rows]
        CheckError -- No --> ExplainError[Explain DB Error in User-friendly terms]
        Synthesize --> FinalResponse[Final Support Response]
        ExplainError --> FinalResponse
    end
```

---

## 3. Comparative Analysis: Basic vs. Advanced Implementations

### Prompt Chaining
- **Basic (`q2_prompt_chaining.py`)**:
  - Prompts are hardcoded for direct topics.
  - Generates three outputs in parallel pipelines without validating if previous steps returned empty or low-quality text.
- **Advanced (`prompt_chaining.py`)**:
  - Uses a modular `generate_step` helper function.
  - Implements strict input length guards (`len(text) < 50`).
  - Performs validation checks on intermediate outputs before proceeding to subsequent steps.
  - Enforces strict constraints in the prompt (e.g., "single cohesive paragraph of no more than 120 words focusing on strategic impact").

### Retrieval-Augmented Generation (RAG)
- **Basic (`q4_rag_qa.py`)**:
  - Chunks text naively without filtering empty blocks.
  - Retrieves only the single most similar chunk ($k=1$).
  - Simple prompt instruction without system behavior constraints.
- **Advanced (`rag_qa.py`)**:
  - Cleans input data, filtering out title markers and empty space.
  - Retrieves the top $k=2$ chunks for a broader context.
  - Prints distance metrics in console for debugging retrieval precision.
  - Adds a negative constraint guard: *"If the answer cannot be found in the context, respond with 'I cannot answer this based on the provided document.'"* to prevent hallucinations.

---

## 4. Key Lessons & Implementation Guidelines

1. **Hallucination Prevention**:
   Strict negative prompts (e.g., "Answer ONLY from the provided context") combined with precise vector retrieval dramatically decrease model confabulations.
2. **Context Contextualization**:
   Chunking files on structural boundaries (like double-newlines or Markdown headers) keeps semantic ideas cohesive compared to naive character or token chunking.
3. **Database Security and Safety**:
   When implementing Text-to-SQL, the LLM should only have read-only access (via database permissions) to prevent SQL injection or accidental data modification.
4. **Intermediate State Validations**:
   In complex chains, checking that intermediate steps succeeded and met length or pattern criteria ensures the robustness of the final output.

---

## 5. Local Setup & Execution Guide

### Prerequisites
Install the required packages in your Python virtual environment:
```bash
pip install google-genai sentence-transformers faiss-cpu numpy
```

### Database Initialization
Before running the Text-to-SQL script, set up the SQLite database:
```bash
python setup_db.py
```

### Running the Workflows
1. **To run the advanced summarization pipeline**:
   ```bash
   python prompt_chaining.py
   ```
2. **To run the grounded QA assistant**:
   ```bash
   python rag_qa.py
   ```
3. **To query the database using natural language**:
   ```bash
   python text_to_sql.py
   ```
