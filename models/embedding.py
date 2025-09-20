from sentence_transformers import SentenceTransformer

# Cache các model đã load
_model_cache = {}

#  = "keepitreal/vietnamese-sbert"
def get_model(model_name: str):
    """Load model theo tên, có cache để nhanh hơn"""
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]

def encode_text(model, text: str):
    """Encode 1 câu thành vector"""
    return model.encode(text).tolist()