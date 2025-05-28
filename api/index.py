from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Cosmas Ngeno Real Estate Assistant is live on Vercel!"}
