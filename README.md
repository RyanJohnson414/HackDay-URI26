# HackDay-URI26

An AI-powered knowledge graph for finding useful connections across documents
from different fields.

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

## Gemini configuration

Copy `backend/.env.example` to `backend/.env` and add your Gemini API key
there. The `.env` file is ignored by Git. Set `GEMINI_MODEL` to the exact Gemma
model identifier supported by the Gemini API and available to your key; do not
commit the key.

## Intended request flow

1. Upload a PDF, text file, or pasted notes.
2. Parse and chunk the source text, then create embeddings.
3. Compare chunks across documents and ask the configured Gemma model to assess
   and explain promising cross-disciplinary connections.
4. Return source passages and connections for the graph and discovery feed.
