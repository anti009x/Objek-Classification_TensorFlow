import uvicorn
from app import app


if __name__ == "__main__":
    uvicorn.run(app, host="https://objek-classificationtensorflow-production.up.railway.app/", port=8000, reload=True)
