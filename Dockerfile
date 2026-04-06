FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose port for HF Spaces
EXPOSE 7860

# Run the FastAPI server (judges run inference.py separately)
CMD ["uvicorn", "env.server:app", "--host", "0.0.0.0", "--port", "7860"]
