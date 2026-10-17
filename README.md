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
On a new empty database, startup inserts two clearly tagged starter examples
into SQLite so the graph loads from the same API and database as user sources.
The frontend labels this starter data; it does not substitute an in-memory graph.

## Gemini configuration

Copy `backend/.env.example` to `backend/.env` and add your Gemini API key
there. The `.env` file is ignored by Git. Set `GEMINI_MODEL` to the exact Gemma
model identifier supported by the Gemini API and available to your key; do not
commit the key.

## Intended request flow

1. Add a PDF, CSV/TSV, text/Markdown file, public web URL, or pasted note.
2. Extract the full text through the document parser, then ask the configured
   `GEMINI_MODEL` to generate grounded titles, domains, tags, and page-based
   section summaries.
3. Chunk and embed the parsed source, compare chunks across documents, and ask
   that same configured model to assess and explain promising connections.
4. Persist source metadata, sections, passages, vectors, topics, and connections
   in SQLite and return them through the graph API.
