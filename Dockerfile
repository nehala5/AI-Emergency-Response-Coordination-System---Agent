FROM python:3.10-slim

WORKDIR /app

COPY . .

# Ensure root files can be imported
ENV PYTHONPATH=/app

RUN pip install --no-cache-dir .

# Hugging Face Spaces port
EXPOSE 7860

CMD ["server"]
