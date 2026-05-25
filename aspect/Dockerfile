# 1. Sử dụng luôn bản Image Java 21 chính thức từ Eclipse Temurin (Ubuntu-based)
FROM eclipse-temurin:21-jdk-noble

# Thiết lập các biến môi trường cho Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 2. Cài đặt Python 3.11 và các công cụ cần thiết qua apt
# Image Temurin bản 'noble' chạy Ubuntu 24.04, ta thêm PPA deadsnakes để cài đúng bản 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Tạo liên kết (symlink) để khi gõ 'python' hoặc 'python3' nó hiểu là bản 3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Kiểm tra phiên bản của cả 2 thằng ngay tại bước build để đảm bảo chuẩn xác
RUN java --version && python3 --version && pip --version

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Sao chép file requirements và cài đặt thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ source code của bạn vào container
COPY . .

# Expose cổng nếu chạy FastAPI
EXPOSE 8000

# Lệnh chạy uvicorn bên trong container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]