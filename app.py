
# ============================================================
# ResearchMind AI
# Evidence-Grounded Intelligence for Academic Research
# ============================================================

import gc
import hashlib
import html
import re
import time
from collections import Counter

import faiss
import numpy as np
import streamlit as st
import torch

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ResearchMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "ResearchMind AI"
APP_TAGLINE = "Evidence-grounded intelligence for academic research"

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

CHUNK_SIZE = 1600
CHUNK_OVERLAP = 180

ASK_TOP_K = 5
ANALYSIS_TOP_K = 5
COMPARISON_TOP_K = 4

MAX_CONTEXT_CHARS = 11500
MIN_CHUNK_LENGTH = 120


# ============================================================
# 2. CUSTOM VISUAL DESIGN
# ============================================================

st.markdown(
    """
<style>

/* ------------------------------------------------------------
   Global
------------------------------------------------------------ */

.block-container {
    max-width: 1450px;
    padding-top: 1.4rem;
    padding-bottom: 4rem;
}

[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128, 128, 128, 0.18);
}

[data-testid="stMetric"] {
    border: 1px solid rgba(128, 128, 128, 0.18);
    border-radius: 16px;
    padding: 18px 20px;
    background: rgba(128, 128, 128, 0.035);
}

[data-testid="stMetricValue"] {
    font-weight: 750;
}

div[data-testid="stExpander"] {
    border-radius: 14px;
    border: 1px solid rgba(128, 128, 128, 0.18);
}

.stButton > button {
    border-radius: 10px;
    font-weight: 650;
    min-height: 2.7rem;
}

.stTextArea textarea {
    border-radius: 12px;
}


/* ------------------------------------------------------------
   Hero
------------------------------------------------------------ */

.rm-hero {
    padding: 2.1rem 2.3rem;
    border-radius: 22px;
    margin-bottom: 1.35rem;
    border: 1px solid rgba(124, 92, 255, 0.25);
    background:
        radial-gradient(
            circle at 90% 10%,
            rgba(124, 92, 255, 0.19),
            transparent 35%
        ),
        linear-gradient(
            135deg,
            rgba(79, 70, 229, 0.12),
            rgba(124, 92, 255, 0.04)
        );
}

.rm-brand {
    font-size: 2.35rem;
    font-weight: 820;
    line-height: 1.1;
    letter-spacing: -0.035em;
    margin-bottom: 0.55rem;
}

.rm-tagline {
    font-size: 1.05rem;
    opacity: 0.78;
    max-width: 800px;
    line-height: 1.65;
}

.rm-badge {
    display: inline-block;
    margin-top: 1rem;
    padding: 0.38rem 0.72rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    border: 1px solid rgba(124, 92, 255, 0.30);
    background: rgba(124, 92, 255, 0.09);
}


/* ------------------------------------------------------------
   Section headings
------------------------------------------------------------ */

.rm-section-title {
    font-size: 1.4rem;
    font-weight: 760;
    margin-bottom: 0.2rem;
}

.rm-section-subtitle {
    opacity: 0.68;
    line-height: 1.6;
    margin-bottom: 1rem;
}


/* ------------------------------------------------------------
   Answer card
------------------------------------------------------------ */

.rm-answer-header {
    padding: 0.8rem 1rem;
    border-radius: 12px;
    margin-top: 1rem;
    margin-bottom: 0.75rem;
    border-left: 4px solid #7c5cff;
    background: rgba(124, 92, 255, 0.07);
    font-weight: 750;
}


/* ------------------------------------------------------------
   Status
------------------------------------------------------------ */

.rm-ready {
    padding: 0.75rem 0.9rem;
    border-radius: 12px;
    border: 1px solid rgba(34, 197, 94, 0.25);
    background: rgba(34, 197, 94, 0.07);
    margin-bottom: 0.8rem;
}

.rm-empty {
    padding: 2rem;
    border-radius: 18px;
    text-align: center;
    border: 1px dashed rgba(128, 128, 128, 0.30);
    background: rgba(128, 128, 128, 0.025);
}


/* ------------------------------------------------------------
   Paper cards
------------------------------------------------------------ */

.rm-paper-card {
    padding: 1rem 1.1rem;
    border-radius: 14px;
    border: 1px solid rgba(128, 128, 128, 0.17);
    margin-bottom: 0.7rem;
    background: rgba(128, 128, 128, 0.025);
}

.rm-paper-name {
    font-weight: 720;
    margin-bottom: 0.25rem;
}

.rm-paper-meta {
    font-size: 0.86rem;
    opacity: 0.67;
}


/* ------------------------------------------------------------
   Footer
------------------------------------------------------------ */

.rm-footer {
    text-align: center;
    opacity: 0.58;
    font-size: 0.82rem;
    margin-top: 2rem;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 3. SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "collection": None,
    "collection_signature": None,
    "processing_time": None,
    "last_answer": None,
    "last_analysis": None,
    "last_comparison": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# 4. AI MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():

    # Use GPU for fast collection embedding.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device=device,
    )

    return model


@st.cache_resource(show_spinner=False)
def load_language_model():

    if torch.cuda.is_available():

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_NAME,
        )

        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

    else:

        tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_NAME,
        )

        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

    model.eval()

    return tokenizer, model


def get_embedding_model():
    return load_embedding_model()


def get_language_model():
    return load_language_model()


# ============================================================
# 5. GENERATION
# ============================================================

def generate_text(
    prompt,
    max_new_tokens=320,
):

    tokenizer, model = get_language_model()

    messages = [
        {
            "role": "system",
            "content": (
                "You are ResearchMind AI, an academic research assistant. "
                "You must reason only from the research evidence supplied "
                "in the prompt. Never invent unsupported facts. "
                "Distinguish the selected paper's own contribution from "
                "related work cited by that paper. When evidence is "
                "insufficient, say so explicitly."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=6500,
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    start = time.perf_counter()

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.perf_counter() - start

    generated = output[
        0,
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    return answer, elapsed


# ============================================================
# 6. TEXT CLEANING
# ============================================================

def clean_pdf_text(text):

    if not text:
        return ""

    # Join common PDF line-break hyphenation:
    # "recom-\nmendation" -> "recommendation"
    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        text,
    )

    text = text.replace("\x00", " ")

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def looks_like_reference_chunk(text):

    lower = text.lower()

    reference_signals = [
        "references",
        "bibliography",
        "doi.org/",
        "proceedings of the",
        "et al., 20",
        "journal of ",
    ]

    score = sum(
        signal in lower
        for signal in reference_signals
    )

    # Stronger bibliography heuristic:
    year_mentions = len(
        re.findall(
            r"\b(?:19|20)\d{2}\b",
            text,
        )
    )

    return (
        score >= 3
        and year_mentions >= 4
    )


# ============================================================
# 7. PDF EXTRACTION
# ============================================================

def extract_pdf_pages(uploaded_file):

    uploaded_file.seek(0)

    reader = PdfReader(
        uploaded_file
    )

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):

        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""

        text = clean_pdf_text(
            raw_text
        )

        if text:

            pages.append(
                {
                    "paper": uploaded_file.name,
                    "page": page_number,
                    "text": text,
                }
            )

    uploaded_file.seek(0)

    return pages


# ============================================================
# 8. RESEARCH CHUNKING
# ============================================================

def create_research_chunks(pages):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )

    chunks = []

    for page in pages:

        page_chunks = splitter.split_text(
            page["text"]
        )

        for chunk_number, text in enumerate(
            page_chunks,
            start=1,
        ):

            text = text.strip()

            if len(text) < MIN_CHUNK_LENGTH:
                continue

            chunks.append(
                {
                    "paper": page["paper"],
                    "page": page["page"],
                    "chunk": chunk_number,
                    "text": text,
                    "is_reference": looks_like_reference_chunk(
                        text
                    ),
                }
            )

    return chunks


# ============================================================
# 9. EMBEDDINGS + FAISS
# ============================================================

def create_vector_index(chunks):

    embedding_model = get_embedding_model()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )

    return embeddings, index


def process_research_collection(
    uploaded_files,
):

    start = time.perf_counter()

    pages = []

    for uploaded_file in uploaded_files:

        pages.extend(
            extract_pdf_pages(
                uploaded_file
            )
        )

    if not pages:
        raise ValueError(
            "No machine-readable text could be extracted "
            "from the uploaded PDF files."
        )

    chunks = create_research_chunks(
        pages
    )

    if not chunks:
        raise ValueError(
            "No usable research chunks could be created."
        )

    embeddings, index = create_vector_index(
        chunks
    )

    paper_names = sorted(
        {
            chunk["paper"]
            for chunk in chunks
        }
    )

    page_counts = Counter(
        page["paper"]
        for page in pages
    )

    chunk_counts = Counter(
        chunk["paper"]
        for chunk in chunks
    )

    reference_chunks = sum(
        chunk["is_reference"]
        for chunk in chunks
    )

    elapsed = time.perf_counter() - start

    return {
        "pages": pages,
        "chunks": chunks,
        "embeddings": embeddings,
        "index": index,
        "paper_names": paper_names,
        "page_counts": dict(page_counts),
        "chunk_counts": dict(chunk_counts),
        "reference_chunks": reference_chunks,
    }, elapsed


# ============================================================
# 10. QUERY EMBEDDING
# ============================================================

def embed_query(query):

    embedding_model = get_embedding_model()

    embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(
        embedding,
        dtype=np.float32,
    )


# ============================================================
# 11. RESULT DEDUPLICATION
# ============================================================

def normalize_for_duplicate_check(text):

    text = text.lower()

    text = re.sub(
        r"\W+",
        " ",
        text,
    )

    return " ".join(
        text.split()
    )


def deduplicate_results(
    results,
    limit,
):

    selected = []
    fingerprints = set()

    for result in results:

        normalized = normalize_for_duplicate_check(
            result["text"]
        )

        fingerprint = normalized[:500]

        if fingerprint in fingerprints:
            continue

        fingerprints.add(
            fingerprint
        )

        selected.append(
            result
        )

        if len(selected) >= limit:
            break

    return selected


# ============================================================
# 12. COLLECTION RETRIEVAL
# ============================================================

def search_collection(
    query,
    collection,
    k=ASK_TOP_K,
    exclude_references=True,
):

    start = time.perf_counter()

    query_embedding = embed_query(
        query
    )

    candidate_k = min(
        max(k * 4, 12),
        len(collection["chunks"]),
    )

    scores, indices = collection[
        "index"
    ].search(
        query_embedding,
        candidate_k,
    )

    candidates = []

    for score, chunk_index in zip(
        scores[0],
        indices[0],
    ):

        if chunk_index < 0:
            continue

        chunk = collection[
            "chunks"
        ][chunk_index]

        if (
            exclude_references
            and chunk["is_reference"]
        ):
            continue

        candidates.append(
            {
                "paper": chunk["paper"],
                "page": chunk["page"],
                "chunk": chunk["chunk"],
                "text": chunk["text"],
                "score": float(score),
            }
        )

    results = deduplicate_results(
        candidates,
        k,
    )

    elapsed = time.perf_counter() - start

    return results, elapsed


# ============================================================
# 13. PAPER-SPECIFIC RETRIEVAL
# ============================================================

def search_specific_paper(
    query,
    paper_name,
    collection,
    k=ANALYSIS_TOP_K,
):

    start = time.perf_counter()

    paper_indices = [
        index
        for index, chunk
        in enumerate(
            collection["chunks"]
        )
        if (
            chunk["paper"] == paper_name
            and not chunk["is_reference"]
        )
    ]

    if not paper_indices:
        return [], 0.0

    query_embedding = embed_query(
        query
    )[0]

    paper_embeddings = collection[
        "embeddings"
    ][paper_indices]

    scores = (
        paper_embeddings
        @ query_embedding
    )

    candidate_count = min(
        max(k * 3, k),
        len(paper_indices),
    )

    positions = np.argsort(
        scores
    )[::-1][:candidate_count]

    candidates = []

    for position in positions:

        chunk_index = paper_indices[
            int(position)
        ]

        chunk = collection[
            "chunks"
        ][chunk_index]

        candidates.append(
            {
                "paper": chunk["paper"],
                "page": chunk["page"],
                "chunk": chunk["chunk"],
                "text": chunk["text"],
                "score": float(
                    scores[position]
                ),
            }
        )

    results = deduplicate_results(
        candidates,
        k,
    )

    elapsed = time.perf_counter() - start

    return results, elapsed


# ============================================================
# 14. CONTEXT CONSTRUCTION
# ============================================================

def build_research_context(
    results,
    max_chars=MAX_CONTEXT_CHARS,
):

    sections = []
    total_chars = 0

    for number, result in enumerate(
        results,
        start=1,
    ):

        section = (
            f"\nSOURCE {number}\n"
            f"Paper: {result['paper']}\n"
            f"Page: {result['page']}\n"
            f"Evidence:\n{result['text']}\n"
        )

        if (
            total_chars + len(section)
            > max_chars
        ):
            break

        sections.append(
            section
        )

        total_chars += len(
            section
        )

    return "\n".join(
        sections
    )


# ============================================================
# 15. RESEARCH Q&A
# ============================================================

def ask_researchmind(
    question,
    collection,
):

    results, retrieval_time = search_collection(
        question,
        collection,
        k=ASK_TOP_K,
    )

    context = build_research_context(
        results
    )

    prompt = f"""
RESEARCH QUESTION:
{question}

RETRIEVED RESEARCH EVIDENCE:
{context}

TASK:
Answer the research question by synthesizing only the evidence above.

REQUIREMENTS:
1. Use only the retrieved evidence.
2. Do not introduce outside knowledge.
3. Do not invent missing information.
4. Prefer synthesis over copying passages.
5. Cite important factual claims using:
   [Paper: filename, Page: number]
6. Every citation must correspond to supplied evidence.
7. If the evidence does not adequately answer the question,
   state that clearly.
8. Write in a clear academic style.
"""

    answer, generation_time = generate_text(
        prompt,
        max_new_tokens=360,
    )

    return {
        "answer": answer,
        "sources": results,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
    }


# ============================================================
# 16. STRUCTURED PAPER ANALYSIS
# ============================================================

ANALYSIS_QUESTIONS = {

    "Research Objective":
        (
            "What main research objective, research problem, "
            "motivation, or research question do the authors "
            "of this paper address?"
        ),

    "Methodology":
        (
            "What methodology, framework, architecture, pipeline, "
            "or system do the authors themselves propose or use?"
        ),

    "Dataset / Data Sources":
        (
            "What datasets, samples, documents, participants, "
            "corpora, job postings, courses, or other data sources "
            "are actually used by the authors in this research?"
        ),

    "Models / Techniques":
        (
            "What models, algorithms, AI techniques, machine "
            "learning methods, NLP methods, embedding approaches, "
            "or computational techniques are actually used by "
            "the authors in their proposed approach?"
        ),

    "Results / Findings":
        (
            "What main experimental results, findings, "
            "contributions, observations, or conclusions are "
            "reported by the authors?"
        ),

    "Limitations / Future Work":
        (
            "What limitations, weaknesses, challenges, threats, "
            "or future research directions are explicitly "
            "identified by the authors?"
        ),
}


ANALYSIS_ICONS = {
    "Research Objective": "🎯",
    "Methodology": "🧪",
    "Dataset / Data Sources": "🗂️",
    "Models / Techniques": "🤖",
    "Results / Findings": "📊",
    "Limitations / Future Work": "🚧",
}


def analyze_paper_aspect(
    paper_name,
    aspect,
    collection,
):

    question = ANALYSIS_QUESTIONS[
        aspect
    ]

    results, retrieval_time = search_specific_paper(
        question,
        paper_name,
        collection,
        k=ANALYSIS_TOP_K,
    )

    context = build_research_context(
        results
    )

    prompt = f"""
SELECTED PAPER:
{paper_name}

RESEARCH ASPECT:
{aspect}

ANALYSIS QUESTION:
{question}

RETRIEVED EVIDENCE:
{context}

TASK:
Produce a focused academic analysis of the selected research aspect.

STRICT REQUIREMENTS:
1. Use only evidence from {paper_name}.
2. Do not use outside knowledge.
3. Describe only methods, datasets, models, results, limitations,
   or contributions that the evidence supports.
4. Distinguish the authors' own work from related work they cite.
5. Do not attribute a cited researcher's method to the selected
   paper unless the evidence explicitly says the authors use it.
6. Cite important claims using:
   [Paper: {paper_name}, Page: number]
7. If the retrieved evidence is insufficient, explicitly say so.
8. Keep the analysis academically precise and concise.
"""

    answer, generation_time = generate_text(
        prompt,
        max_new_tokens=360,
    )

    return {
        "answer": answer,
        "sources": results,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
    }


# ============================================================
# 17. FULL PAPER ANALYSIS
# ============================================================

def analyze_full_paper(
    paper_name,
    collection,
    progress_callback=None,
):

    analyses = {}

    aspects = list(
        ANALYSIS_QUESTIONS.keys()
    )

    for index, aspect in enumerate(
        aspects,
        start=1,
    ):

        if progress_callback:
            progress_callback(
                index,
                len(aspects),
                aspect,
            )

        analyses[aspect] = (
            analyze_paper_aspect(
                paper_name,
                aspect,
                collection,
            )
        )

    return analyses


# ============================================================
# 18. PAPER COMPARISON
# ============================================================

def compare_research_papers(
    paper_1,
    paper_2,
    aspect,
    collection,
):

    question = ANALYSIS_QUESTIONS[
        aspect
    ]

    results_1, retrieval_time_1 = (
        search_specific_paper(
            question,
            paper_1,
            collection,
            k=COMPARISON_TOP_K,
        )
    )

    results_2, retrieval_time_2 = (
        search_specific_paper(
            question,
            paper_2,
            collection,
            k=COMPARISON_TOP_K,
        )
    )

    context_1 = build_research_context(
        results_1,
        max_chars=6000,
    )

    context_2 = build_research_context(
        results_2,
        max_chars=6000,
    )

    prompt = f"""
COMPARISON ASPECT:
{aspect}

PAPER 1:
{paper_1}

EVIDENCE FOR PAPER 1:
{context_1}

PAPER 2:
{paper_2}

EVIDENCE FOR PAPER 2:
{context_2}

TASK:
Directly compare the two papers with respect to {aspect}.

STRICT REQUIREMENTS:
1. Use only the supplied evidence.
2. Do not introduce outside knowledge.
3. Do not invent similarities or differences.
4. Distinguish each paper's own work from related work it cites.
5. Cite important claims using:
   [Paper: filename, Page: number]
6. If a comparison cannot be supported by the evidence,
   explicitly say so.
7. Focus on meaningful research differences rather than
   superficial wording differences.

USE THIS STRUCTURE:

### {paper_1}
Explain the evidence relevant to {aspect}.

### {paper_2}
Explain the evidence relevant to {aspect}.

### Key Similarities
Compare supported similarities directly.

### Key Differences
Compare supported differences directly.

### Research Interpretation
Give a short evidence-grounded interpretation of how the
approaches relate or differ.
"""

    answer, generation_time = generate_text(
        prompt,
        max_new_tokens=480,
    )

    return {
        "answer": answer,
        "paper_1_sources": results_1,
        "paper_2_sources": results_2,
        "retrieval_time": (
            retrieval_time_1
            + retrieval_time_2
        ),
        "generation_time": generation_time,
    }


# ============================================================
# 19. FILE SIGNATURE
# ============================================================

def uploaded_files_signature(
    files,
):

    signatures = []

    for file in files:

        file.seek(0)

        content = file.read()

        file.seek(0)

        digest = hashlib.sha256(
            content
        ).hexdigest()

        signatures.append(
            (
                file.name,
                len(content),
                digest,
            )
        )

    return tuple(
        signatures
    )


# ============================================================
# 20. DOWNLOADABLE REPORTS
# ============================================================

def build_analysis_report(
    paper_name,
    analyses,
):

    lines = [
        "# ResearchMind AI — Research Paper Analysis",
        "",
        f"## Paper",
        paper_name,
        "",
    ]

    for aspect, result in analyses.items():

        lines.extend(
            [
                f"## {aspect}",
                "",
                result["answer"],
                "",
            ]
        )

    lines.extend(
        [
            "---",
            "Generated by ResearchMind AI",
        ]
    )

    return "\n".join(
        lines
    )


def build_comparison_report(
    paper_1,
    paper_2,
    aspect,
    comparison,
):

    return f"""# ResearchMind AI — Paper Comparison

## Papers

- {paper_1}
- {paper_2}

## Comparison Aspect

{aspect}

## Research Comparison

{comparison["answer"]}

---

Generated by ResearchMind AI
"""


# ============================================================
# 21. UI HELPERS
# ============================================================

def safe_html(text):
    return html.escape(
        str(text)
    )


def render_section_header(
    title,
    subtitle,
):

    st.markdown(
        f"""
<div class="rm-section-title">{safe_html(title)}</div>
<div class="rm-section-subtitle">{safe_html(subtitle)}</div>
""",
        unsafe_allow_html=True,
    )


def render_sources(
    results,
    title="Supporting Evidence",
):

    if not results:

        st.info(
            "No supporting evidence was retrieved."
        )

        return

    st.markdown(
        f"### 📚 {title}"
    )

    for number, result in enumerate(
        results,
        start=1,
    ):

        label = (
            f"Source {number} · "
            f"{result['paper']} · "
            f"Page {result['page']} · "
            f"Similarity {result['score']:.3f}"
        )

        with st.expander(
            label,
            expanded=False,
        ):

            st.caption(
                f"Paper: {result['paper']}  •  "
                f"Page: {result['page']}  •  "
                f"Chunk: {result['chunk']}  •  "
                f"Semantic similarity: {result['score']:.4f}"
            )

            st.write(
                result["text"]
            )


def render_performance(
    retrieval_time,
    generation_time,
):

    total = (
        retrieval_time
        + generation_time
    )

    col1, col2, col3 = st.columns(
        3
    )

    col1.metric(
        "Evidence Retrieval",
        f"{retrieval_time:.2f}s",
    )

    col2.metric(
        "AI Generation",
        f"{generation_time:.2f}s",
    )

    col3.metric(
        "Total Response",
        f"{total:.2f}s",
    )


def clear_collection():

    st.session_state.collection = None
    st.session_state.collection_signature = None
    st.session_state.processing_time = None
    st.session_state.last_answer = None
    st.session_state.last_analysis = None
    st.session_state.last_comparison = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 22. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🧠 ResearchMind AI"
    )

    st.caption(
        "Your evidence-grounded research workspace"
    )

    st.divider()

    st.markdown(
        "### 📂 Research Library"
    )

    uploaded_files = st.file_uploader(
        "Upload research papers",
        type=["pdf"],
        accept_multiple_files=True,
        help=(
            "Upload machine-readable academic PDF files. "
            "Scanned image-only PDFs may require OCR."
        ),
    )

    if uploaded_files:

        st.caption(
            f"{len(uploaded_files)} paper(s) selected"
        )

        for uploaded_file in uploaded_files:

            size_mb = (
                uploaded_file.size
                / 1024
                / 1024
            )

            st.markdown(
                f"📄 **{uploaded_file.name}**  \n"
                f"<small>{size_mb:.2f} MB</small>",
                unsafe_allow_html=True,
            )

        current_signature = (
            uploaded_files_signature(
                uploaded_files
            )
        )

        collection_changed = (
            st.session_state.collection is None
            or current_signature
            != st.session_state.collection_signature
        )

        if collection_changed:

            if st.button(
                "⚡ Build Research Index",
                type="primary",
                use_container_width=True,
            ):

                progress = st.progress(
                    0,
                    text="Preparing research collection..."
                )

                try:

                    progress.progress(
                        15,
                        text="Extracting PDF text..."
                    )

                    with st.spinner(
                        "Building the ResearchMind semantic index..."
                    ):

                        collection, processing_time = (
                            process_research_collection(
                                uploaded_files
                            )
                        )

                    progress.progress(
                        100,
                        text="Research index ready."
                    )

                    st.session_state.collection = (
                        collection
                    )

                    st.session_state.collection_signature = (
                        current_signature
                    )

                    st.session_state.processing_time = (
                        processing_time
                    )

                    st.success(
                        "Research collection indexed."
                    )

                    time.sleep(
                        0.4
                    )

                    st.rerun()

                except Exception as error:

                    st.error(
                        f"Research processing failed: {error}"
                    )

        else:

            st.markdown(
                """
<div class="rm-ready">
<strong>● Research index ready</strong><br>
<small>Your papers are ready for AI research analysis.</small>
</div>
""",
                unsafe_allow_html=True,
            )

    else:

        st.info(
            "Upload PDF papers to create your research workspace."
        )

    if (
        st.session_state.collection
        is not None
    ):

        st.divider()

        collection = (
            st.session_state.collection
        )

        st.markdown(
            "### ⚙️ Workspace"
        )

        if st.session_state.processing_time is not None:

            st.caption(
                f"Index built in "
                f"{st.session_state.processing_time:.2f}s"
            )

        if st.button(
            "🗑️ Clear Research Collection",
            use_container_width=True,
        ):

            clear_collection()

            st.rerun()

    st.divider()

    with st.expander(
        "ℹ️ AI System"
    ):

        st.write(
            "**Embedding:** BGE-M3"
        )

        st.write(
            "**Retrieval:** FAISS"
        )

        st.write(
            "**Generator:** Qwen2.5-3B-Instruct"
        )

        st.write(
            "**Quantization:** 4-bit"
        )

        st.write(
            "**Architecture:** Retrieval-Augmented Generation"
        )

        if torch.cuda.is_available():

            st.write(
                f"**GPU:** "
                f"{torch.cuda.get_device_name(0)}"
            )

        else:

            st.write(
                "**Compute:** CPU"
            )


# ============================================================
# 23. HERO
# ============================================================

st.markdown(
    """
<div class="rm-hero">
    <div class="rm-brand">🧠 ResearchMind AI</div>
    <div class="rm-tagline">
        Understand research. Compare evidence. Discover insights.
        ResearchMind AI transforms academic papers into an
        evidence-grounded research workspace powered by semantic
        retrieval and generative AI.
    </div>
    <div class="rm-badge">
        Evidence-Grounded RAG · Paper & Page Citations
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 24. COLLECTION DASHBOARD
# ============================================================

collection = (
    st.session_state.collection
)

if collection is not None:

    metric_1, metric_2, metric_3, metric_4 = (
        st.columns(4)
    )

    metric_1.metric(
        "📚 Research Papers",
        len(
            collection["paper_names"]
        ),
    )

    metric_2.metric(
        "📄 Pages Indexed",
        len(
            collection["pages"]
        ),
    )

    metric_3.metric(
        "🧩 Evidence Chunks",
        len(
            collection["chunks"]
        ),
    )

    metric_4.metric(
        "⚡ Research Status",
        "Ready",
    )

else:

    st.markdown(
        """
<div class="rm-empty">
    <h3>Build your research workspace</h3>
    <p>
        Upload academic PDF papers from the sidebar and build
        the semantic research index to activate ResearchMind AI.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )


st.write("")


# ============================================================
# 25. MAIN NAVIGATION
# ============================================================

ask_tab, analyze_tab, compare_tab, library_tab = (
    st.tabs(
        [
            "💬 Ask Research",
            "🔬 Analyze Paper",
            "⚖️ Compare Papers",
            "📚 Research Library",
        ]
    )
)


# ============================================================
# 26. ASK RESEARCH
# ============================================================

with ask_tab:

    render_section_header(
        "Ask Your Research Collection",
        (
            "Ask evidence-grounded questions across every "
            "paper in your research workspace."
        ),
    )

    if collection is None:

        st.warning(
            "Upload and index research papers first."
        )

    else:

        st.markdown(
            "**Try asking:**"
        )

        example_col_1, example_col_2, example_col_3 = (
            st.columns(3)
        )

        with example_col_1:

            st.caption(
                "💡 What are the main approaches proposed across these papers?"
            )

        with example_col_2:

            st.caption(
                "💡 What datasets and AI techniques are used?"
            )

        with example_col_3:

            st.caption(
                "💡 What research limitations appear across the collection?"
            )

        question = st.text_area(
            "Research question",
            placeholder=(
                "Ask ResearchMind anything about your uploaded "
                "research papers..."
            ),
            height=120,
            key="research_question",
        )

        ask_button = st.button(
            "🧠 Ask ResearchMind",
            type="primary",
            use_container_width=True,
            key="ask_research_button",
        )

        if ask_button:

            if not question.strip():

                st.warning(
                    "Enter a research question first."
                )

            else:

                try:

                    with st.spinner(
                        "ResearchMind is retrieving evidence "
                        "and reasoning over your papers..."
                    ):

                        result = ask_researchmind(
                            question.strip(),
                            collection,
                        )

                    st.session_state.last_answer = (
                        result
                    )

                except Exception as error:

                    st.error(
                        f"Research query failed: {error}"
                    )

        if (
            st.session_state.last_answer
            is not None
        ):

            result = (
                st.session_state.last_answer
            )

            st.markdown(
                """
<div class="rm-answer-header">
🧠 ResearchMind Answer
</div>
""",
                unsafe_allow_html=True,
            )

            st.markdown(
                result["answer"]
            )

            with st.expander(
                "⚡ Response Performance"
            ):

                render_performance(
                    result[
                        "retrieval_time"
                    ],
                    result[
                        "generation_time"
                    ],
                )

            render_sources(
                result["sources"]
            )


# ============================================================
# 27. ANALYZE PAPER
# ============================================================

with analyze_tab:

    render_section_header(
        "Analyze an Individual Study",
        (
            "Examine a paper's research objective, methodology, "
            "data, models, results, and limitations using "
            "paper-specific evidence."
        ),
    )

    if collection is None:

        st.warning(
            "Upload and index research papers first."
        )

    else:

        selector_col_1, selector_col_2 = (
            st.columns([1.1, 1])
        )

        with selector_col_1:

            selected_paper = st.selectbox(
                "Research paper",
                collection["paper_names"],
                key="analysis_paper",
            )

        with selector_col_2:

            selected_aspect = st.selectbox(
                "Research aspect",
                list(
                    ANALYSIS_QUESTIONS.keys()
                ),
                key="analysis_aspect",
            )

        action_col_1, action_col_2 = (
            st.columns(2)
        )

        with action_col_1:

            analyze_one = st.button(
                "🔬 Analyze Selected Aspect",
                type="primary",
                use_container_width=True,
            )

        with action_col_2:

            analyze_all = st.button(
                "📑 Generate Full Paper Analysis",
                use_container_width=True,
            )

        if analyze_one:

            try:

                with st.spinner(
                    f"Analyzing {selected_aspect.lower()}..."
                ):

                    result = analyze_paper_aspect(
                        selected_paper,
                        selected_aspect,
                        collection,
                    )

                st.session_state.last_analysis = {
                    "type": "single",
                    "paper": selected_paper,
                    "aspect": selected_aspect,
                    "result": result,
                }

            except Exception as error:

                st.error(
                    f"Paper analysis failed: {error}"
                )

        if analyze_all:

            progress_bar = st.progress(
                0,
                text="Starting full paper analysis..."
            )

            def update_analysis_progress(
                current,
                total,
                aspect,
            ):

                percentage = int(
                    current
                    / total
                    * 100
                )

                progress_bar.progress(
                    percentage,
                    text=f"Analyzing: {aspect}",
                )

            try:

                analyses = analyze_full_paper(
                    selected_paper,
                    collection,
                    progress_callback=update_analysis_progress,
                )

                st.session_state.last_analysis = {
                    "type": "full",
                    "paper": selected_paper,
                    "analyses": analyses,
                }

                progress_bar.progress(
                    100,
                    text="Full research analysis complete."
                )

            except Exception as error:

                st.error(
                    f"Full paper analysis failed: {error}"
                )

        analysis_state = (
            st.session_state.last_analysis
        )

        if analysis_state is not None:

            if (
                analysis_state["type"]
                == "single"
            ):

                result = analysis_state[
                    "result"
                ]

                aspect = analysis_state[
                    "aspect"
                ]

                icon = ANALYSIS_ICONS.get(
                    aspect,
                    "🔬",
                )

                st.markdown(
                    f"""
<div class="rm-answer-header">
{icon} {safe_html(aspect)}
</div>
""",
                    unsafe_allow_html=True,
                )

                st.caption(
                    analysis_state["paper"]
                )

                st.markdown(
                    result["answer"]
                )

                with st.expander(
                    "⚡ Analysis Performance"
                ):

                    render_performance(
                        result[
                            "retrieval_time"
                        ],
                        result[
                            "generation_time"
                        ],
                    )

                render_sources(
                    result["sources"]
                )

            else:

                st.markdown(
                    """
<div class="rm-answer-header">
📑 Full Research Paper Analysis
</div>
""",
                    unsafe_allow_html=True,
                )

                st.caption(
                    analysis_state["paper"]
                )

                for aspect, result in (
                    analysis_state[
                        "analyses"
                    ].items()
                ):

                    icon = ANALYSIS_ICONS.get(
                        aspect,
                        "🔬",
                    )

                    st.markdown(
                        f"### {icon} {aspect}"
                    )

                    st.markdown(
                        result["answer"]
                    )

                    with st.expander(
                        f"Evidence for {aspect}"
                    ):

                        render_sources(
                            result["sources"],
                            title="Retrieved Evidence",
                        )

                    st.divider()

                report = build_analysis_report(
                    analysis_state["paper"],
                    analysis_state[
                        "analyses"
                    ],
                )

                st.download_button(
                    "⬇️ Download Analysis Report",
                    data=report,
                    file_name=(
                        f"{analysis_state['paper']}"
                        "_ResearchMind_Analysis.md"
                    ),
                    mime="text/markdown",
                    use_container_width=True,
                )


# ============================================================
# 28. COMPARE PAPERS
# ============================================================

with compare_tab:

    render_section_header(
        "Compare Research Papers",
        (
            "Retrieve evidence independently from two studies "
            "and compare their research approaches directly."
        ),
    )

    if collection is None:

        st.warning(
            "Upload and index research papers first."
        )

    elif len(
        collection["paper_names"]
    ) < 2:

        st.warning(
            "At least two papers are required for comparison."
        )

    else:

        compare_col_1, compare_col_2 = (
            st.columns(2)
        )

        with compare_col_1:

            paper_1 = st.selectbox(
                "Paper 1",
                collection["paper_names"],
                key="compare_paper_1",
            )

        with compare_col_2:

            paper_2 = st.selectbox(
                "Paper 2",
                collection["paper_names"],
                index=1,
                key="compare_paper_2",
            )

        comparison_aspect = st.selectbox(
            "Comparison aspect",
            list(
                ANALYSIS_QUESTIONS.keys()
            ),
            key="compare_aspect",
        )

        compare_button = st.button(
            "⚖️ Compare Research",
            type="primary",
            use_container_width=True,
        )

        if compare_button:

            if paper_1 == paper_2:

                st.warning(
                    "Choose two different research papers."
                )

            else:

                try:

                    with st.spinner(
                        "ResearchMind is retrieving evidence "
                        "from both papers and comparing them..."
                    ):

                        result = (
                            compare_research_papers(
                                paper_1,
                                paper_2,
                                comparison_aspect,
                                collection,
                            )
                        )

                    st.session_state.last_comparison = {
                        "paper_1": paper_1,
                        "paper_2": paper_2,
                        "aspect": comparison_aspect,
                        "result": result,
                    }

                except Exception as error:

                    st.error(
                        f"Paper comparison failed: {error}"
                    )

        comparison_state = (
            st.session_state.last_comparison
        )

        if comparison_state is not None:

            result = comparison_state[
                "result"
            ]

            st.markdown(
                """
<div class="rm-answer-header">
⚖️ ResearchMind Comparison
</div>
""",
                unsafe_allow_html=True,
            )

            st.caption(
                f"{comparison_state['paper_1']}  ↔  "
                f"{comparison_state['paper_2']}  •  "
                f"{comparison_state['aspect']}"
            )

            st.markdown(
                result["answer"]
            )

            with st.expander(
                "⚡ Comparison Performance"
            ):

                render_performance(
                    result[
                        "retrieval_time"
                    ],
                    result[
                        "generation_time"
                    ],
                )

            st.markdown(
                "### 📚 Comparison Evidence"
            )

            evidence_tab_1, evidence_tab_2 = (
                st.tabs(
                    [
                        comparison_state[
                            "paper_1"
                        ],
                        comparison_state[
                            "paper_2"
                        ],
                    ]
                )
            )

            with evidence_tab_1:

                render_sources(
                    result[
                        "paper_1_sources"
                    ],
                    title=(
                        f"Evidence from "
                        f"{comparison_state['paper_1']}"
                    ),
                )

            with evidence_tab_2:

                render_sources(
                    result[
                        "paper_2_sources"
                    ],
                    title=(
                        f"Evidence from "
                        f"{comparison_state['paper_2']}"
                    ),
                )

            comparison_report = (
                build_comparison_report(
                    comparison_state[
                        "paper_1"
                    ],
                    comparison_state[
                        "paper_2"
                    ],
                    comparison_state[
                        "aspect"
                    ],
                    result,
                )
            )

            st.download_button(
                "⬇️ Download Comparison Report",
                data=comparison_report,
                file_name="ResearchMind_Comparison.md",
                mime="text/markdown",
                use_container_width=True,
            )


# ============================================================
# 29. RESEARCH LIBRARY
# ============================================================

with library_tab:

    render_section_header(
        "Research Library",
        (
            "Inspect the academic collection currently indexed "
            "inside your ResearchMind workspace."
        ),
    )

    if collection is None:

        st.warning(
            "Upload and index research papers first."
        )

    else:

        overview_1, overview_2, overview_3 = (
            st.columns(3)
        )

        overview_1.metric(
            "Total Papers",
            len(
                collection["paper_names"]
            ),
        )

        overview_2.metric(
            "Total Pages",
            len(
                collection["pages"]
            ),
        )

        overview_3.metric(
            "Searchable Evidence",
            (
                len(
                    collection["chunks"]
                )
                - collection[
                    "reference_chunks"
                ]
            ),
        )

        st.write("")

        for paper in collection[
            "paper_names"
        ]:

            pages = collection[
                "page_counts"
            ].get(
                paper,
                0,
            )

            chunks = collection[
                "chunk_counts"
            ].get(
                paper,
                0,
            )

            st.markdown(
                f"""
<div class="rm-paper-card">
    <div class="rm-paper-name">📄 {safe_html(paper)}</div>
    <div class="rm-paper-meta">
        {pages} pages · {chunks} semantic research chunks
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown(
            "### 🔎 Research Index"
        )

        st.write(
            "ResearchMind creates one semantic vector index "
            "for the uploaded collection. Paper analysis and "
            "comparison reuse these embeddings instead of "
            "reprocessing each PDF."
        )

        st.markdown(
            "### 🛡️ Evidence Grounding"
        )

        st.write(
            "Generated responses are instructed to use only "
            "retrieved evidence. Each important research claim "
            "should be linked back to the originating paper "
            "and page."
        )


# ============================================================
# 30. FOOTER
# ============================================================

st.markdown(
    """
<div class="rm-footer">
    🧠 ResearchMind AI · Evidence-Grounded Academic Intelligence
    <br>
    BGE-M3 · FAISS · Qwen2.5 · 4-bit Quantization · RAG
</div>
""",
    unsafe_allow_html=True,
)
