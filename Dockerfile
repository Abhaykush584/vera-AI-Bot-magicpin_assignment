FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY bot.py .

# Render/Railway/Fly set PORT env var automatically
ENV PORT=8080
EXPOSE 8080

CMD ["python", "bot.py"]
