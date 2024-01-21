from fastapi import FastAPI
from inference import LLMClient


app = FastAPI()

llm = LLMClient()


@app.get("/")
async def ping():
    return {'status': 'ok'}


@app.get("/inference")
async def inference():
    return llm()
