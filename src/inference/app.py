from fastapi import FastAPI
from inference import InferenceHandler


MODEL_PATH = "models/llama-2-7b-chat.Q2_K.gguf"


app = FastAPI()

llm = InferenceHandler(MODEL_PATH).wait_until_server_up()


@app.get("/")
async def ping():
    return {'status': 'ok'}


@app.get("/inference")
async def inference():
    return llm()
