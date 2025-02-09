# RAG-Powered Document CLI QA Chatbot

Question-answering system that combines retrieval-augmented generation (RAG) with hybrid vector search capabilities.
Additionally router query classifier has been added.

## Features

- **Hybrid Search**: Combines dense and sparse vectors for enhanced retrieval accuracy
- **PDF Processing**: Automatic Markdown aware ingestion of PDF documents from Downloads folder
- **Context Classification**: BERT-based classifier determines when retrieval is needed
- **Milvus Integration**: Scalable vector database for efficient similarity search
- **Streaming Responses**: Real-time responses with message history context

## Usage

1. **Before lauching enusre you have:**
- Docker Milvus instance ([https://milvus.io/docs/install_standalone-windows.md])
- PDF files in ~App/Downloads directory for PDF documents ingestion
- Api key from [https://glhf.chat] to use LLM models (It has free credits)
- ModernBERT pretrained model [https://drive.google.com/drive/folders/18nMrZimkgTbqsT2gr7eFHHEdS6oXTQCb?usp=sharing]


2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Ingest PDF documents** (place PDFs in ~/Downloads first)
```bash
python App/main.py --ingest
```

4. **Start chatbot interface**
```bash
python App/main.py
```

## Additional information

- Basic BERT training script is placed in Notebooks folder
