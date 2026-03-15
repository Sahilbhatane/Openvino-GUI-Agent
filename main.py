import uvicorn
from config import API_HOST, API_PORT

if __name__ == "__main__":
    print(f"Starting GUI Agent server on http://{API_HOST}:{API_PORT}")
    print("POST /run-task  -- submit a natural-language GUI task")
    print("GET  /health    -- check if the model is loaded\n")
    uvicorn.run("api.server:app", host=API_HOST, port=API_PORT, reload=False)
