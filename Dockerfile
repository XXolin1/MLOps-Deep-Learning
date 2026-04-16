FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make run_API.sh executable
RUN chmod +x run_API.sh

# Expose port
EXPOSE 8000

# Run the application using the run_API.sh script
CMD ["bash", "run_API.sh"]
