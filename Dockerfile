FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the precomputed model file
COPY model.pkl .
COPY app.py .

EXPOSE 5000

CMD ["python", "app.py"]