# Consilience

Consilience means the convergence of knowledge across different fields. This
AI-powered knowledge graph finds useful connections across your documents and
surfaces ideas that bridge disciplines.

## MVP structure

```text
HackDay-URI26/
├── README.md
├── .gitignore
├── backend/
│   ├── .env.example
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py
│   │   ├── core/config.py
│   │   ├── api/routes/
│   │   │   ├── documents.py
│   │   │   ├── graph.py
│   │   │   └── discoveries.py
│   │   ├── db/
│   │   │   ├── models.py
│   │   │   └── session.py
│   │   ├── schemas/
│   │   └── services/
│   │       ├── document_parser.py
│   │       ├── chunking.py
│   │       ├── embeddings.py
│   │       ├── gemini_client.py
│   │       └── connection_engine.py
│   └── tests/
├── frontend/
│   ├── package.json
│   ├── index.html
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── api/client.ts
│       ├── components/
│       │   ├── UploadPanel.tsx
│       │   ├── KnowledgeGraph.tsx
│       │   └── DiscoveryFeed.tsx
│       └── types/index.ts
└── data/
    └── uploads/
```

## Backend setup (Linux / WSL)

Run these commands from the `backend` directory. Create the virtual environment
once, then activate it and install the project dependencies:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
uvicorn app.main:app --reload
```

If the `venv` creation command reports that `ensurepip` is unavailable, install
the matching `python3-venv` system package first. Each new terminal needs the
`source .venv/bin/activate` command before running the backend.

## Backend setup (Windows PowerShell)

Run these commands from the repository root. Create and install the virtual
environment once:

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Start the backend in that terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal, start the website with `cd frontend` and `npm run dev`.
Keep both terminals running. The backend initializes SQLite and starts serving
before loading the optional context-check files in the background, so model
provider delays do not prevent the website from connecting.

## Database and graph API

The default database is `backend/knowledge_graph.db`. Relative SQLite paths in
`DATABASE_URL` are resolved from the backend directory, regardless of the
directory from which the server is launched. FastAPI creates the database
tables during application startup (`app.main` lifespan); no separate
migration/setup command is required for a new local database.

The frontend requests `GET /api/graph`; Vite proxies `/api` to the backend at
`http://127.0.0.1:8000`. The graph response is read from the SQLite documents,
chunks, topic, and connection tables. Ingesting a new source persists its
passages, discovers cross-document links, and refreshes the graph response.
At startup, the backend reads every `.txt` file in
`backend/context_check_sources/` and sends it through the same parser, Gemma
analysis, embedding, topic grouping, and connection discovery pipeline as an
uploaded file. The folder contains only source prose: no expected labels,
vectors, topic assignments, or connection scores. Unchanged fixtures are not
re-ingested on later starts; edited source text is reprocessed so new evidence
reaches the model, and failed or partial fixtures are retried. A working Gemini
API key and network connection are needed to process these files. Provider
errors are logged, and the backend still starts.

## Gemini configuration

Copy `backend/.env.example` to `backend/.env` and add your Gemini API key
there. The `.env` file is ignored by Git. Set `GEMINI_MODEL` to the exact Gemma
model identifier supported by the Gemini API and available to your key; do not
commit the key.

## Intended request flow

1. On startup, read the context-check text corpus; or add a PDF, CSV/TSV,
   text/Markdown file, public web URL, or pasted note.
2. Extract the full text through the document parser, then ask the configured
   `GEMINI_MODEL` to generate grounded titles, domains, tags, and page-based
   section summaries.
3. Chunk and embed the parsed source, group comparable passages into topics
   (using analyzed source titles for single-source groups), and ask the
   configured model to assess cross-document candidates with embedding
   similarity of at least 0.01. Retain model-confirmed links with strength of
   at least 0.50; default candidate and result counts are not capped.
4. Persist source metadata, sections, passages, vectors, uniquely named topics,
   and Gemma-confirmed connections in SQLite and return them through the graph
   API. Each connection stores vector similarity, model strength, blended weight,
   a semantic-distance label (local / related / distant), and discipline scope
   (within-discipline / cross-discipline / unknown). The semantic-distance label
   is derived from blended weight: at least 0.80 is local, at least 0.60 is
   related, and lower confirmed weights are distant. Existing SQLite databases
   are migrated and existing connections are backfilled at startup without
   deleting source records.
