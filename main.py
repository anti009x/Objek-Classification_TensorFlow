import uvicorn
from app import app


if __name__ == "__main__":
    #use this code for deploy
    #uvicorn.run(app, host="127.0.0.0")
    #use this code for local running
    uvicorn.run(app, host="127.0.0.0", port=8000, reload=True)

