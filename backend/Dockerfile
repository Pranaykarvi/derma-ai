# Use a minimal base image with Python 3.10
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install OS-level dependencies (you can add more if needed)
RUN apt-get update && apt-get install -y git && apt-get clean

# Copy everything into the container's /app folder
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the default port Hugging Face Spaces expects (Gradio and FastAPI default)
EXPOSE 7860

# Run your FastAPI app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
