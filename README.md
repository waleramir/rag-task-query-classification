# RAG-Powered Document QA Chatbot

Question-answering system that combines retrieval-augmented generation (RAG) with hybrid vector search capabilities.

## Features

- **Hybrid Search**: Combines dense and sparse vectors for enhanced retrieval accuracy
- **PDF Processing**: Automatic ingestion of PDF documents from Downloads folder
- **Context Classification**: BERT-based classifier determines when retrieval is needed
- **Milvus Integration**: Scalable vector database for efficient similarity search
- **Streaming Responses**: Real-time responses with message history context

## Installation

```bash
pip install pymilvus transformers python-dotenv langchain pymupdf4llm mirascope-openai
```

## Usage

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Ingest PDF documents** (place PDFs in ~/Downloads first)
```bash
python App/main.py --ingest
```

3. **Start chatbot interface**
```bash
python App/main.py
```

## Configuration

```python
# Key components:
- Hybrid encoder (BGE-M3) for document embeddings
- Milvus vector database for storage/retrieval
- ModernBERT classifier for query analysis
- GLHF LLM models (via Mirascope framework) for response generation
```

## Dependencies

- Requires docker Milvus instance (automatic setup included)
- Uses ~/Downloads directory for PDF ingestion
- Api key from [https://glhf.chat]
- ModernBERT pretrained model [https://drive.google.com/drive/folders/18nMrZimkgTbqsT2gr7eFHHEdS6oXTQCb?usp=sharing]