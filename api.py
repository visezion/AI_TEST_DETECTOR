from fastapi import FastAPI
from infer import detect_text

app = FastAPI()


@app.get("/")
def root():
    return {"message": "AI Text Detector running"}


@app.post("/detect")
def detect_api(data: dict):
    text = data.get("text", "")
    result = detect_text(text)
    return result
