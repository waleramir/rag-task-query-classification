from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    WeightedRanker,
)


def setup_milvus():
    # Milvus lite does not work on Windows btw, use docker container and its uri in that case
    # connections.connect(uri="./milvus.db")
    connections.connect(uri="http://localhost:19530")
    col_name = "hybrid_col"
    if utility.has_collection(col_name):
        # dropping the same name collection
        # Collection(col_name).drop()
        col = Collection(col_name)
        # print(col)
        print("Collection loaded!")
    else:
        fields = [
            FieldSchema(
                name="pk",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=True,
                max_length=100,
            ),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(
                name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024
            ),  # BGE-M3 dense dimension
        ]

        schema = CollectionSchema(fields)

        if utility.has_collection(col_name):
            print("dropping the same name collection")
            Collection(col_name).drop()

        col = Collection(col_name, schema, consistency_level="Strong")

        col.create_index(
            "sparse_vector",
            {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"},
        )

        col.create_index(
            "dense_vector", {"index_type": "AUTOINDEX", "metric_type": "IP"}
        )

        col.load()
    return col


def insert_batch(col, texts, embeds, batch_size=50):
    """Insert documents in batches with their embeddings"""
    total_inserted = 0

    print(f"len(texts)={len(texts)}")
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_dense = embeds["dense"][i : i + batch_size]
        batch_sparse = embeds["sparse"][i : i + batch_size]
        entities = [batch_texts, batch_sparse, batch_dense]

        col.insert(entities)
        total_inserted += len(batch_texts)

    return total_inserted


def hybrid_search(
    col, query_dense, query_sparse, sparse_weight=0.7, dense_weight=1.0, limit=10
):
    dense_req = AnnSearchRequest(
        data=[query_dense],
        anns_field="dense_vector",
        param={"metric_type": "IP"},
        limit=limit,
    )

    sparse_req = AnnSearchRequest(
        data=[query_sparse],
        anns_field="sparse_vector",
        param={"metric_type": "IP"},
        limit=limit,
    )

    rerank = WeightedRanker(sparse_weight, dense_weight)
    res = col.hybrid_search(
        [sparse_req, dense_req], rerank=rerank, limit=limit, output_fields=["text"]
    )[0]
    return [hit.get("text") for hit in res]
