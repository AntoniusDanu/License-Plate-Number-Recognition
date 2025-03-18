# Gunakan Python 3.10 sebagai base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy semua file ke dalam container
COPY . .

# Install semua dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Jalankan aplikasi FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
