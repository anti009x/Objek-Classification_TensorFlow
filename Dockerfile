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

#Running Docker File Here !
#Please Readme cant change port and host
CMD ["uvicorn", "main:app", "--host=127.0.0.0"]


