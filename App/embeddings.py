from milvus_model.hybrid import BGEM3EmbeddingFunction


class BGEHybridEncoder:
    def __init__(self):
        self.model = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
        self.dense_dim = self.model.dim["dense"]

    def encode(self, text: str) -> dict:
        embeddings = self.model([text])
        return {"dense": embeddings["dense"][0], "sparse": embeddings["sparse"][0]}
