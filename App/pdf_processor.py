from pymupdf4llm import to_markdown
from langchain.text_splitter import MarkdownTextSplitter
from pathlib import Path


def process_pdfs(docs_dir="Documents"):
    splitter = MarkdownTextSplitter(
        chunk_size=512, chunk_overlap=64, strip_whitespace=True
    )

    for pdf_path in Path(docs_dir).glob("*.pdf"):
        md_text = to_markdown(str(pdf_path))
        chunks = splitter.create_documents([md_text])

        for chunk in chunks:
            yield {
                "content": chunk.page_content,
                "metadata": {
                    "source": pdf_path.name,
                    "page": chunk.metadata.get("page", 0),
                },
            }
