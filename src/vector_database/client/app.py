from fastapi import FastAPI
from chroma_client import VectorDB
import pandas as pd
from pydantic import BaseModel
from typing import Dict, List


class Query(BaseModel):
    question: str


app = FastAPI()

data = pd.read_csv("data/nhs_mental_health_data.csv")

vector_db_client = VectorDB(data).wait_until_server_up().build_documents().build_collections()


@app.get("/")
async def ping():
    return {'status': 'ok'}


@app.post("/query")
async def create_context(payload: Query) -> Dict[str, str]:
    vector_db_client.query(payload.question)
    return {"message": "context created successfully"}


@app.get("/question")
async def read_question() -> str:
    return vector_db_client.question


@app.get("/documents")
async def read_documents() -> List[str]:
    return vector_db_client.context_documents


@app.get("/urls")
async def read_urls() -> List[str]:
    return vector_db_client.context_urls


@app.get("/titles")
async def read_titles() -> List[str]:
    return vector_db_client.context_titles
