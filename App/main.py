from mirascope.core import BaseMessageParam, openai, prompt_template
from transformers import pipeline, AutoTokenizer

from pymilvus import WeightedRanker
import torch
from openai import OpenAI
from dotenv import load_dotenv
import argparse
import os
import sys

# Local imports
from pdf_processor import process_pdfs
from milvus_model.hybrid import BGEM3EmbeddingFunction
from milvus_client import insert_batch, hybrid_search, setup_milvus

load_dotenv()

# Using glhf.chat as model provider
client = OpenAI(
    api_key=os.environ.get("GLHF_API_KEY"),
    base_url="https://glhf.chat/api/openai/v1",
)


class Chatbot:
    """Simple RAG CLI chatbot with document retrieval capabilities"""

    history: list[BaseMessageParam] = []
    ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
    classifier: pipeline = pipeline(
        "text-classification",
        model="best_rag_classifier",
        tokenizer=AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base"),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    collection = setup_milvus()

    def _needs_retrieval(self, question: str) -> bool:
        """Determine if a question requires document retrieval"""
        result = self.classifier(question)[0]
        print(result)
        return result["label"] == "Needs_Retrieval"

    def _get_context(self, question: str) -> str:
        """Retrieve relevant context from Milvus database."""
        embeddings = self.ef([question])
        results = hybrid_search(
            self.collection,
            embeddings["dense"][0],
            embeddings["sparse"]._getrow(0),
            sparse_weight=0.7,
            dense_weight=1.0,
        )
        return "\n".join(chunk for chunk in results)

    @openai.call("hf:qwen/Qwen2.5-Coder-32B-Instruct", client=client, stream=True)
    @prompt_template(
        """
        SYSTEM: You are an AI assistant. You are able to find answers to the questions from the document contextual passage snippets provided.
        If no context provided, say you can't provide answer to documents questions and then proceed chat in normal manner.
        Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
        MESSAGES: {self.history}
        USER:
        <context>
        {context}
        </context>
        <question>
        {question}
        </question>
        """
    )
    def _call(self, question: str, context: str = "") -> None:
        """Generate response using LLM with retrieved context."""
        ...

    def run(self) -> None:
        """Main chat interface loop."""
        print("Chatbot initialized. Type 'exit' to quit.")
        print("Assistant: I can answer your questions about documents!")
        while True:
            question = input("\nUser: ").strip()
            if question.lower() in ["exit", "quit"]:
                print("Assistant: Have a great day!")
                break

            context = (
                self._get_context(question) if self._needs_retrieval(question) else ""
            )
            # print(context)
            print("Assistant: ", end="", flush=True)
            stream = self._call(question, context)

            for chunk, _ in stream:
                print(chunk.content, end="", flush=True)
            print()

            if stream.user_message_param:
                self.history.append(stream.user_message_param)
            self.history.append(stream.message_param)

    @classmethod
    def ingest_documents(cls) -> None:
        """Process PDFs and load embeddings into Milvus."""
        print("Ingesting documents from Downloads folder...")
        # Process all documents first
        chunks = list(process_pdfs())
        print("All Documents processed.")

        # Prepare batch data
        texts = [chunk["content"] for chunk in chunks]
        # print(texts)
        embeds = cls.ef(texts)
        # dense_embeds = [ef.encode(doc["content"])["dense"] for doc in docs]

        # Insert all documents in batches
        total_inserted = insert_batch(cls.collection, texts, embeds)
        print(f"Inserted {total_inserted} documents")
        print("Ingestion completed successfully.")


if __name__ == "__main__":
    # Command line interface setup
    parser = argparse.ArgumentParser(
        description="RAG Chatbot - Document QA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Process PDFs from Downloads folder and load into Milvus",
    )
    args = parser.parse_args()

    if args.ingest:
        Chatbot.ingest_documents()
        sys.exit(0)

    Chatbot().run()
