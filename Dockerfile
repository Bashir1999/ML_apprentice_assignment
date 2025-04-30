# Base Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all necessary files
COPY task_1.py task_2.py task_4.py requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command
CMD ["python3"]
