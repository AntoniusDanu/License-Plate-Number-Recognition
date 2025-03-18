# Gunakan image Python dengan OpenCV & Torch
FROM python:3.9

# Set work directory
WORKDIR /app

# Copy semua file ke container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (sesuai framework)
EXPOSE 8000

# Jalankan aplikasi
CMD ["python", "app.py"]
