
# 🚀 [Tips Hindawi](https://tipshindawi.com/) Internship (August–October) 2026

> 🎓 This project was built during the [**Tips Hindawi**](https://tipshindawi.com/) **Internship (August–October) 2026**.

## 👤 Participant

| Field | Value |
| --- | --- |
| Full Name | Abdelrahman Mohamed |
| Project Name | ResearchMind AI |
| GitHub Username | *Abdalrahman01-CS* |
| Internship Batch | August–October 2026 |
| Training Program | Large Language Models (LLMs) Program |
| Organization | [**Edrak for AI**](https://edrak4ai.com/en) |

---

# 📖 Project Overview

**ResearchMind AI** is an AI-powered research paper assistant designed to make academic literature easier to explore, analyze, and compare.

The system allows users to upload multiple academic research papers in PDF format and interact with them through an evidence-grounded **Retrieval-Augmented Generation (RAG)** pipeline.

Instead of relying only on the language model's internal knowledge, ResearchMind AI retrieves relevant evidence directly from the uploaded papers before generating an answer.

The system combines semantic search using **BGE-M3 embeddings**, efficient vector retrieval with **FAISS**, and answer generation using a **4-bit quantized Qwen2.5-3B-Instruct** large language model.

ResearchMind AI provides three main research workflows:

- 💬 Ask questions across a collection of research papers.
- 🔬 Analyze an individual research paper by research aspect.
- ⚖️ Compare multiple research papers using retrieved evidence.

The project is delivered through an interactive **Streamlit** interface.

---

# ✨ Features

### 💬 Ask Papers

Users can ask natural-language questions about their uploaded research papers.

ResearchMind AI:

- Searches the research collection using semantic retrieval.
- Identifies the most relevant passages.
- Generates an evidence-grounded answer.
- Displays the supporting paper names and page numbers.

### 🔬 Analyze Paper

Users can select an individual paper and analyze important research aspects such as:

- Research objective
- Methodology
- Dataset / data sources
- Models and techniques
- Results and findings
- Limitations and future work

Relevant evidence is retrieved specifically from the selected paper before the analysis is generated.

### ⚖️ Compare Papers

ResearchMind AI can retrieve evidence from different papers and generate a structured academic comparison.

This helps identify:

- Similarities between research approaches
- Differences in methodology
- Different models and techniques
- Research contributions
- Limitations and research gaps

### 🔎 Semantic Research Retrieval

Research paper content is transformed into dense vector representations using **BGE-M3**.

The system uses **FAISS** similarity search to efficiently retrieve research passages that are semantically related to the user's question.

### 📚 Multi-Paper Research Collection

Multiple PDFs can be uploaded and processed as a single research collection while retaining metadata such as:

- Paper filename
- Page number
- Research text

### 📑 Evidence-Grounded Responses

Generated answers are based on retrieved research evidence rather than unrestricted model generation.

Where applicable, responses reference the relevant paper and page to make the generated analysis easier to verify.

### ⚡ Optimized Local LLM Inference

The generation model uses **4-bit quantization**, reducing GPU memory requirements and making it possible to run the system efficiently on limited GPU resources.

---

# 🛠️ Technologies Used

| Technology | Purpose |
| --- | --- |
| Python | Core programming language |
| Streamlit | Interactive web application |
| Qwen2.5-3B-Instruct | Large Language Model for research answer generation |
| BGE-M3 | Semantic embedding model |
| FAISS | Vector similarity search |
| Hugging Face Transformers | LLM loading and inference |
| Sentence Transformers | Embedding generation |
| PyTorch | Deep-learning backend |
| BitsAndBytes | 4-bit model quantization |
| PyPDF | PDF text extraction |
| LangChain Text Splitters | Research document chunking |
| Hugging Face Hub | Model access and authentication |

---

# ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Abdalrahman01-CS/ResearchMind-AI.git
cd ResearchMind-AI
```

### 2. Install the required dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Hugging Face access

A Hugging Face account and access token may be required to download the models used by the application.

For security, API tokens and credentials should be stored as environment variables or secrets and should **never be committed to GitHub**.

### 4. Run ResearchMind AI

```bash
streamlit run app.py
```

Streamlit will start the ResearchMind AI interface.

---

# 🚀 Usage

### Step 1 — Launch ResearchMind AI

Start the Streamlit application.

### Step 2 — Upload Research Papers

Upload one or more academic research papers in PDF format.

ResearchMind AI extracts the text, divides it into research chunks, generates embeddings, and creates the searchable research collection.

### Step 3 — Choose a Research Tool

Use one of the three main workflows:

**💬 Ask Papers**

Enter a research question to retrieve relevant evidence and generate an answer grounded in the uploaded papers.

**🔬 Analyze Paper**

Select a paper and research aspect to generate structured analysis based on evidence from that specific paper.

**⚖️ Compare Papers**

Select research papers and a comparison aspect to generate an evidence-based comparison.

### Step 4 — Review Supporting Evidence

ResearchMind AI displays the retrieved sources so that generated claims can be traced back to the uploaded research papers.

---

# 📈 Results

ResearchMind AI successfully demonstrates an end-to-end **Retrieval-Augmented Generation system for academic research assistance**.

The completed system can:

- Process multiple academic PDF documents.
- Convert research content into semantic embeddings.
- Store and search research vectors using FAISS.
- Retrieve evidence relevant to research questions.
- Generate answers grounded in retrieved passages.
- Analyze individual research papers.
- Compare research papers using retrieved evidence.
- Preserve paper and page metadata for research traceability.
- Provide all functionality through an interactive Streamlit interface.

The project demonstrates how **Large Language Models, semantic embeddings, vector search, document processing, and RAG** can be integrated into a practical academic research application.

---

# 🔮 Future Improvements

ResearchMind AI can be extended with several additional capabilities:

- **Hybrid Retrieval and Reranking:** Combine dense semantic retrieval with keyword-based retrieval and reranking models for more precise evidence selection.

- **Research Library Persistence:** Store processed research collections so users do not need to reprocess papers when restarting the application.

- **Advanced Citation Verification:** Automatically verify generated statements against retrieved passages and provide more granular citation references.

- **Research Paper Summarization:** Generate structured summaries covering objectives, methodology, datasets, experiments, results, limitations, and contributions.

- **Larger Research Collections:** Extend the retrieval architecture to efficiently handle hundreds or thousands of research papers.

- **Research Gap Discovery:** Compare findings across papers to assist researchers in identifying limitations, conflicting findings, and potential research gaps.

---

# 📚 About the Internship

This project was developed as part of the [**Tips Hindawi**](https://tipshindawi.com/) **Internship (August–October 2026)**, and it will be showcased on the official [Tips Hindawi](https://tipshindawi.com/) website.

[Tips Hindawi](https://tipshindawi.com/) is the internships department of [**Edrak for AI**](https://edrak4ai.com/en), and the internship encourages participants to build real-world projects, apply practical skills, and showcase their work through GitHub.

For more information about the internship, training programs, and upcoming batches, visit the official [Tips Hindawi](https://tipshindawi.com/) website.

---

# 📄 License

This project is shared for educational and portfolio purposes.
