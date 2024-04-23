FROM python:3.11.5

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

# Install Python dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
    
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000"]


