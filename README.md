"# demo" 

## Giới thiệu
Ứng dụng demo tìm kiếm ngữ nghĩa tiếng việt sử dụng Qdrant, Sentence Transformers và Flask.

## Yêu cầu
- Python 3.8+
- Docker (để chạy Qdrant)
- pip

## Hướng dẫn cài đặt và chạy


### 1. Tạo và kích hoạt virtual environment (tuỳ chọn)
```bash
cd your-repo
python -m venv venv
venv\Scripts\activate  # Windows
# hoặc
source venv/bin/activate  # Linux/Mac
```

### 2. Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```
> Nếu chưa có file `requirements.txt`, bạn có thể cài đặt thủ công:
> ```
> pip install flask python-dotenv sentence-transformers qdrant-client
> ```

### 3. Khởi động Qdrant bằng Docker
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 4. Tạo file `.env` và đặt secret key
Tạo file `.env` trong thư mục gốc với nội dung:
```
APP_SECRET_KEY=your_secret_key_here
```

### 5. Chuẩn bị dữ liệu
- Dữ liệu mẫu đã có sẵn trong `data/data.txt`.
- Bạn có thể chỉnh sửa hoặc mở rộng file này theo ý muốn.

### 6. Index dữ liệu vào Qdrant
```bash
python db/index_data.py
```

### 7. Chạy ứng dụng Flask
```bash
python app.py
```
- Truy cập [http://localhost:5000] trên trình duyệt để sử dụng demo.

## Ghi chú
- Nếu muốn thay đổi mô hình embedding, chỉnh sửa file `db/index_data.py` và `models/embedding.py`.
- Để cập nhật dữ liệu mới, hãy sửa `data/data.txt` và chạy lại bước 6.