FROM python:3.14-slim

# Hugging Face runs Docker Spaces as user 1000.
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
USER user
ENV HOME=/home/user

EXPOSE 8501
CMD ["streamlit", "run", "Frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
