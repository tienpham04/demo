import os
from dotenv import load_dotenv
import time
from flask import Flask, render_template, request, jsonify, session
from qdrant_client import QdrantClient
from models.embedding import get_model, encode_text
from db.vector_db import search_in_qdrant
from sentence_transformers import CrossEncoder

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("APP_SECRET_KEY")

# Kết nối Qdrant server (Docker)
qdrant = QdrantClient(host="localhost", port=6333)

# Map model → collection
COLLECTIONS = {
    "paraphrase-multilingual-MiniLM-L12-v2": "demo_paraphrase-multilingual-MiniLM-L12-v2",
    "keepitreal/vietnamese-sbert": "demo_keepitreal_vietnamese-sbert",
    "BAAI/bge-m3": "demo_BAAI_bge-m3",
}

# Khởi tạo CrossEncoder (chỉ cần load 1 lần)
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    encode_time = None
    search_time = None
    history = session.get("history", [])
    if request.method == "POST":
        query = request.form["query"]
        model_name = request.form["model"]

        # 1. Lấy model từ cache (embedding.py)
        model = get_model(model_name)
        t0 = time.time()

        # 2. Encode query
        query_vector = encode_text(model, query)
        encode_time = (time.time() - t0) * 1000  # ms

        # 3. Xác định collection tương ứng
        collection_name = COLLECTIONS[model_name]

        # 4. Search trong Qdrant
        t1 = time.time()
        search_result = search_in_qdrant(qdrant, collection_name, query_vector, top_k=10)  # lấy nhiều hơn để rerank
        search_time = (time.time() - t1) * 1000  # ms

        # 5. Chuẩn bị kết quả hiển thị
        results = [
            {"text": hit.payload.get("text"), "score": hit.score}
            for hit in search_result
        ]

        # 6. Rerank kết quả bằng CrossEncoder
        pairs = [(query, item["text"]) for item in results]
        rerank_scores = cross_encoder.predict(pairs)
        for i, item in enumerate(results):
            item["rerank_score"] = float(rerank_scores[i])
        # Sắp xếp lại theo rerank_score
        results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        # Chỉ lấy top 5 kết quả cuối cùng
        results = results[:5]

        # Lưu vào lịch sử
        history.append({
            "query": query,
            "model": model_name,
            "results": results,
            "encode_time": encode_time,
            "search_time": search_time
        })
        # Giới hạn lịch sử 10 lần gần nhất
        history = history[-10:]
        session["history"] = history

    return render_template(
        "index.html",
        results=results,
        encode_time=encode_time,
        search_time=search_time,
        history=history
    )

@app.route("/search", methods=["POST"])
def api_search():
    """API cho frontend/React hoặc Postman"""
    data = request.json
    query = data.get("query", "")
    model_name = data.get("model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    model = get_model(model_name)
    query_vector = encode_text(model, query)

    collection_name = COLLECTIONS[model_name]

    search_result = search_in_qdrant(qdrant, collection_name, query_vector, limit=5)

    results = [
        {"text": hit.payload.get("text"), "score": hit.score}
        for hit in search_result
    ]
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)
