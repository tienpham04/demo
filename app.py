from flask import Flask, render_template, request
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

# ----------------------------
# 1. Khởi tạo Flask app
# ----------------------------
app = Flask(__name__)

# ----------------------------
# 2. Khởi tạo model + Qdrant
# ----------------------------
model = SentenceTransformer("keepitreal/vietnamese-sbert")  # Model embedding tiếng Việt
qdrant = QdrantClient(":memory:")  # Dùng Qdrant chạy trong RAM (demo)

collection_name = "demo_collection"

# ----------------------------
# 3. Tạo collection
# ----------------------------
qdrant.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

# ----------------------------
# 4. Thêm dữ liệu mẫu
# ----------------------------
documents = [
    "Hà Nội là thủ đô của Việt Nam.",
    "Huế nổi tiếng với các di tích lịch sử.",
    "TP Hồ Chí Minh là trung tâm kinh tế lớn nhất.",
]

vectors = [model.encode(doc).tolist() for doc in documents]

qdrant.upsert(
    collection_name=collection_name,
    points=[
        PointStruct(id=idx, vector=vector, payload={"text": doc})
        for idx, (vector, doc) in enumerate(zip(vectors, documents))
    ],
)

# ----------------------------
# 5. Route chính
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        query = request.form["query"]
        query_vector = model.encode(query).tolist()

        # Tìm kiếm trong Qdrant
        search_result = qdrant.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=3,
        )

        # Lấy text + score từ ScoredPoint
        results = [
            f"{hit.payload.get('text')} (score: {hit.score:.4f})"
            for hit in search_result
        ]

    return render_template("index.html", results=results)


# ----------------------------
# 6. Chạy Flask app
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
