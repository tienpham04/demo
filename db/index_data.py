from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import os

DATA_PATH = os.path.join("data", "data.txt")

# Các model sẽ test
MODELS = {
    "paraphrase-multilingual-MiniLM-L12-v2": 384,
    "keepitreal/vietnamese-sbert": 768,
    "BAAI/bge-m3": 1024,
}

# ======================
# Kết nối Qdrant
# ======================
qdrant = QdrantClient(host="localhost", port=6333)

# ======================
# Đọc dữ liệu từ file
# ======================
with open(DATA_PATH, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

# ======================
# Index dữ liệu cho từng model
# ======================
for model_name, dim in MODELS.items():
    print(f"\n🔹 Đang xử lý model: {model_name} (dim={dim})")

    model = SentenceTransformer(model_name)
    embeddings = model.encode(lines)

    # Tên collection đặt theo model
    collection_name = f"demo_{model_name.replace('/', '_')}"

    # Xóa collection cũ và tạo mới (dùng cách mới, tránh warning)
    if qdrant.collection_exists(collection_name=collection_name):
        qdrant.delete_collection(collection_name=collection_name)
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    # Tạo các PointStruct
    points = [
        PointStruct(id=i, vector=embeddings[i].tolist(), payload={"text": lines[i]})
        for i in range(len(lines))
    ]

    # Upsert vào Qdrant
    qdrant.upsert(collection_name=collection_name, points=points)

    print(f"✅ Đã index {len(lines)} dòng dữ liệu vào collection '{collection_name}'")

print("\n🎉 Hoàn tất indexing cho tất cả model!")