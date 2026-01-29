# Retrieval-Augmented Generation (RAG) with User Docs

The `inference/rag.py` helper lets you create per-workspace knowledge bases, ingest PDFs/Docs, and
retrieve passages with source citations so you can ground your prompts.

## Install dependencies

```shell
pip install -r inference/requirements.txt -r inference/requirements-rag.txt
```

## Ingest documents

Ingest documents into a workspace (each workspace writes its own `kb.json` store):

```shell
python inference/rag.py --workspace acme ingest docs/handbook.pdf docs/faq.docx --tags "handbook,faq"
```

## Query the workspace

Retrieve top passages with source citations:

```shell
python inference/rag.py --workspace acme query "How do I reset my account?" --top-k 3
```
