# Consilience

Consilience is a local-first knowledge graph for exploring how ideas in
different documents relate. Add notes, files, or public web pages; the backend
extracts and analyzes their text, groups similar passages into topics, and asks
the configured Gemini model (Gemma by default) to check possible links between
sources. The website stores the resulting graph in local JSON and source files,
then displays it in a React network view. Website uploads do not write to SQLite.

The app does not have a prewritten list of expected connections. A connection
is created only after the model assesses two real passages and the result
passes the score threshold described below. Embeddings help find candidate
passages, but are not by themselves treated as proof that two sources are
related.

## What happens when you add a source

1. **Read the source.** The backend accepts PDF, CSV/TSV, plain-text or Markdown
   uploads, pasted notes, and public web URLs. PDFs retain page numbers;
   delimited files are rendered as labeled rows; text and notes are treated as
   one page. Unsupported or unreadable sources return an error.
2. **Analyze the text.** The configured Gemini model is asked to derive a title,
   academic or practical domain, tags, and section summaries from the source.
   This analysis is grounded in the supplied text, not an external document
   catalog.
3. **Split and embed passages.** Text is divided into chunks of about 1,200
   characters, with up to 180 characters of overlap. Each chunk is sent to the
   configured embedding model and stored locally with its vector and source location.
   Overlap helps keep ideas near chunk boundaries available to later steps.
4. **Build topic groups.** The backend compares chunk vectors throughout the
   stored corpus. Similar chunks are grouped, then named from a document title
   for single-source groups or informative keywords for multi-source groups.
5. **Check cross-document candidates.** The connection engine compares chunks
   from different documents. It sends promising candidates, along with document
   context such as title, domain, tags, and analyzed sections, to the configured
   model for a grounded relatedness assessment. The assessment includes an
   explanation and short supporting quotes when available.
6. **Persist and display.** Documents, chunks, embeddings, topics, topic
   relations, and accepted connections are stored in a local JSON graph file.
   Uploaded originals are kept in a local folder. The frontend reads the graph
   through the backend API; it does not calculate link scores itself.

## How the math works

There are several scores in the graph. They answer different questions and
should not be read as interchangeable probabilities.

### 1. Embedding cosine similarity

An embedding represents a chunk as a vector of numbers. For two vectors
\(\mathbf{a}\) and \(\mathbf{b}\), the backend calculates cosine similarity:

\[
\operatorname{cosine}(\mathbf{a}, \mathbf{b}) =
\frac{\mathbf{a}\cdot\mathbf{b}}
{\|\mathbf{a}\|\,\|\mathbf{b}\|}
\]

Values closer to 1 indicate vectors pointing in similar directions; values
around 0 indicate little directional similarity. This is a retrieval signal:
it helps choose which passages are worth asking the language model to compare.
It is not the final connection decision.

For an upload, pairs with similarity of at least **0.01** are candidates. The
engine also sends up to the five nearest candidates for each new chunk, even if
their similarity is below that threshold, so a low embedding score alone does
not automatically hide a possible connection. If vectors cannot be compared
because their dimensions differ, shared-word overlap is used to rank the
fallback candidates; incompatible vectors are not given a fabricated cosine
score.

### 2. Model-assessed connection strength

For each candidate pair, the configured Gemini model scores the relationship
from **0 to 1**, using this guidance:

| Model strength | Intended interpretation |
| --- | --- |
| 0.00–0.19 | No meaningful relation |
| 0.20–0.39 | Broad vocabulary or weak analogy |
| 0.40–0.49 | Weak or speculative relation |
| 0.50–0.79 | Specific, evidence-backed shared concept or mechanism |
| 0.80–1.00 | Direct, substantial relation supported by both passages |

The model is instructed to mark a pair relevant only at **0.50 or above**.
Pairs below that threshold are discarded, not saved as connections. Its score
is a model judgment guided by this rubric—not a statistically calibrated
probability that the passages are objectively related.

### 3. Blended connection weight

For a model-approved pair, the displayed and stored weight combines the
embedding signal and model judgment:

\[
\text{weight} =
\operatorname{clamp}_{[0,1]}
\left(0.35 \times \text{cosine similarity}
+ 0.65 \times \text{model strength}\right)
\]

The model strength has more influence (65%) than vector similarity (35%).
Clamping keeps the result between 0 and 1. For example, if similarity is 0.70
and model strength is 0.80, the weight is
\(0.35(0.70)+0.65(0.80)=0.765\), or about **77%** in the interface. This
percentage is a blended ranking score, not a probability or a guarantee.

Accepted connections are labeled by that weight:

| Weight | Semantic-distance label |
| --- | --- |
| 0.80–1.00 | `local` |
| 0.60–0.79 | `related` |
| Below 0.60 | `distant` |

The minimum model strength still applies: a pair is not accepted just because
its embedding similarity is high.

### 4. Topic groups and related-topic links

Topic grouping uses embeddings separately from the model-judged source
connections:

- Two chunks are joined when their cosine similarity is at least **0.72**.
  Connected components form groups, so a chain of similar chunks may place two
  chunks in the same group even if those two chunks are not directly similar
  enough to meet the threshold.
- A topic centroid is the component-wise average of its member chunk vectors.
- Two distinct topics get a related-topic link when their centroid cosine
  similarity is at least **0.58**.

These thresholds are practical grouping rules, not universal definitions of a
topic. They can be tuned in `backend/app/services/topic_service.py`.

For multi-source groups, the topic label is based on up to three informative
keywords. Each word is ranked using a TF-IDF-like weight:

\[
\text{term frequency in group}
\times
\log\left(\frac{\text{number of chunks}+1}
{\text{chunks containing the term}+1}\right)
\]

The frequency factor favors words that occur in the group; the logarithmic
factor favors words that are less common across the corpus. Common stop words
are excluded. If all chunks in a group come from one document, the analyzed
document title is used instead. Duplicate labels receive a numeric suffix.

### 5. Discipline scope

Discipline scope is derived from the analyzed document domains, not from vector
math:

- `within_discipline`: normalized domain strings match.
- `cross_discipline`: both domains are present and differ.
- `unknown`: either domain is missing or marked uncategorized.

This is a simple domain-label comparison. It does not attempt to infer a
hierarchy between disciplines.

## Run locally

You need Python 3.11 or newer, Node.js with npm, and a Gemini API key for
document analysis, embeddings, and model-checked connections. Install the
backend once:

### Windows PowerShell

From the repository root:

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Start the backend in that terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local Vite address printed by `npm run dev` (usually
`http://localhost:5173`). Keep both terminals running: the website sends API
requests through Vite to the backend at `http://127.0.0.1:8000`. If the backend
is temporarily unavailable, the site retries graph loading automatically.

### Linux / WSL

From the repository root, install the backend once:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Start it in one terminal:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Start the frontend in another terminal:

```bash
cd frontend
npm install
npm run dev
```

If creating the Linux virtual environment reports that `ensurepip` is
unavailable, install the matching system `python3-venv` package first.

## Configure Gemini

Copy `backend/.env.example` to `backend/.env`, then set the API key and (if
needed) model names:

```dotenv
GEMINI_API_KEY=your-key
GEMINI_MODEL=gemma-3-27b-it
EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIMENSIONS=768
```

The `.env` file is ignored by Git. Use a model identifier supported by the
Gemini API and enabled for your key; never commit the key. Provider errors can
interrupt analysis or candidate assessment. If one candidate fails, the
ingestion process records a warning and continues with other candidates. If
document analysis or embedding fails, that source may not reach graph
connection assessment.

## Where source data goes

The graph and uploaded source files stay on this machine, and the website talks
to the local backend. However, this is not an offline-only
application: source text is sent to the configured Gemini API for document
analysis and embeddings, and passage text plus document context is sent for
connection assessment. Do not submit material unless using that provider is
appropriate for it. The local JSON graph stores extracted text, source metadata,
embeddings, and assessed connection explanations/evidence.

## Local demo files and API

The website's graph is stored in `backend/local_demo_state.json`, and uploaded
files are copied to `backend/local_demo_sources/`. Both are local to the machine
and ignored by Git. The checked-in `backend/local_demo_seed.json` provides the
initial demo graph. The website backend does not initialize or write to
`knowledge_graph.db`.

Source uploads still run through the configured document parser, embedding
model, topic grouping, and Gemma connection checks. The JSON store keeps the
resulting vectors and evidence so later uploads can be compared against the
same local corpus. File uploads are limited to 10 MB for the demo.

Useful endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Check that the backend is responding |
| `GET /api/local/graph` | Load the local JSON graph used by the website |
| `POST /api/local/documents/upload` | Save and analyze a supported file locally |
| `POST /api/local/documents/url` | Fetch, analyze, and save a public web page locally |
| `POST /api/local/documents/note` | Analyze and save a pasted note locally |

The interactive API documentation is available at `http://127.0.0.1:8000/docs`
when the backend is running.

## Project layout

```text
backend/
  app/
    api/                 HTTP routes and graph serialization
    core/                Environment and application settings
    db/                  Legacy SQLite models and session helpers
    services/local_demo_store.py  Local JSON graph and source file store
    services/             Parsing, chunking, embeddings, Gemini, grouping, links
  context_check_sources/  Plain-text source examples
  tests/                  Backend tests
frontend/
  src/
    api/                  Browser-to-backend client
    components/            Upload, graph, and discovery UI
    App.tsx                Main application view
```

Run backend tests from `backend` with:

```bash
python -m pytest -q
```

Build and type-check the frontend from `frontend` with:

```bash
npm run build
```
