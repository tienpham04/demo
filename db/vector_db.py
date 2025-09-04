from qdrant_client import QdrantClient

# Khởi tạo Qdrant 
qdrant = QdrantClient(host="localhost", port=6333) 

def search_in_qdrant(qdrant_client: QdrantClient, collection_name: str, query_vector, top_k: int = 5):
    """
    Tìm top-k trong collection Qdrant.
    - qdrant_client: kết nối Qdrant
    - collection_name: tên collection (phụ thuộc model)
    - query_vector: vector query
    - top_k: số kết quả cần lấy
    """
    return qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
    )
