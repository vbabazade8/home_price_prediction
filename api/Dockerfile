FROM python:3.14-slim
WORKDIR /app
COPY api/requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY api/ ./api/
COPY models/ ./models/
COPY static/ ./static/
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
