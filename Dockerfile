# Use python:3.10-slim as base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application directories into container
COPY src /app/src
COPY app /app/app
COPY data /app/data
COPY models /app/models
COPY dashboard /app/dashboard
COPY .streamlit /app/.streamlit

# Expose Streamlit port
EXPOSE 8501

# Set default command to run Streamlit app
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
