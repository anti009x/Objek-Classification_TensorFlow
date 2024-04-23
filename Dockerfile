FROM python:3.11.5

WORKDIR /app
COPY . /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Use a script or chain commands to start your application
# For example, if you need to run a FastAPI app with Uvicorn and also have a Python script:
# uvicorn main:app --host=0.0.0.0 --port=8000 --reload
# Note: Changed --host to 0.0.0.0 to allow connections from outside the container

# If you still need to run the Python script, consider using a startup script that runs both commands or choose one as the entry point.
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--PORT=8000", "--reload"]