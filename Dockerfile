FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y default-jre git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN git clone --depth 1 https://github.com/vncorenlp/VnCoreNLP.git /app/vncorenlp

COPY api/ ./api/
COPY src/ ./src/
COPY configs/ ./configs/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]