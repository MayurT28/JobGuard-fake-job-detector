FROM python:3.10-slim

WORKDIR /app

ENV HF_HUB_DISABLE_XET=1
ENV HF_HOME=/root/.cache/huggingface

COPY . .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Download model once during build
RUN python -c "from transformers import BertTokenizer, BertForSequenceClassification; \
    BertTokenizer.from_pretrained('MayurT28/jobguard-bert'); \
    BertForSequenceClassification.from_pretrained('MayurT28/jobguard-bert')"

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]