import json
from app import get_model, encode_text, COLLECTIONS, qdrant, search_in_qdrant

K = 5  # hoặc 10

with open("ground_truth.json", "r", encoding="utf-8") as f:
    gt_data = json.load(f)

models = list(COLLECTIONS.keys())
recall_scores = {model: [] for model in models}

for model_name in models:
    print(f"\nĐánh giá model: {model_name}")
    model = get_model(model_name)
    collection_name = COLLECTIONS[model_name]
    for item in gt_data:
        query = item["query"]
        relevant = set(item["relevant"])
        query_vector = encode_text(model, query)
        results = search_in_qdrant(qdrant, collection_name, query_vector, top_k=K)
        retrieved = set(hit.payload.get("text") for hit in results)
        # Tính recall@k cho truy vấn này
        hit_count = len(relevant & retrieved)
        recall = hit_count / len(relevant) if relevant else 0
        recall_scores[model_name].append(recall)

# In kết quả trung bình
for model_name in models:
    avg_recall = sum(recall_scores[model_name]) / len(recall_scores[model_name])
    print(f"Recall@{K} của {model_name}: {avg_recall:.2f}")