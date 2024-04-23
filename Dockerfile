FROM python:3.11.5

# WORKDIR /app
# COPY . /app

# Install Python dependencies
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
    
CMD ["uvicorn", "main:app", "--host=127.0.0.1", "--port=8000", "--reload"]

