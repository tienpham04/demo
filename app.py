from flask import Flask, render_template, request, jsonify
from qdrant_client import QdrantClient
from models.embedding import get_model, encode_text
from db.vector_db import search_in_qdrant

app = Flask(__name__)

# Kết nối Qdrant server (Docker)
qdrant = QdrantClient(host="localhost", port=6333)

# Map model → collection
COLLECTIONS = {
    "paraphrase-multilingual-MiniLM-L12-v2": "demo_paraphrase-multilingual-MiniLM-L12-v2",
    "keepitreal/vietnamese-sbert": "demo_keepitreal_vietnamese-sbert",
}

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        query = request.form["query"]
        model_name = request.form["model"]

        # 1. Lấy model từ cache (embedding.py)
        model = get_model(model_name)

        # 2. Encode query
        query_vector = encode_text(model, query)

        # 3. Xác định collection tương ứng
        collection_name = COLLECTIONS[model_name]

        # 4. Search trong Qdrant
        search_result = search_in_qdrant(qdrant, collection_name, query_vector, top_k=5)

        # 5. Chuẩn bị kết quả hiển thị
        results = [
            {"text": hit.payload.get("text"), "score": hit.score}
            for hit in search_result
        ]

    return render_template("index.html", results=results)


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
