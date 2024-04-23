FROM python:3.11.5

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

CMD ["uvicorn", "main:app", "--host=127.0.0.1", "--port=8000", "--reload"]
