from llama_cpp import Llama
import requests
import time
from typing import Optional
from typing import Dict, List
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def ping():
    return {'status': 'ok'}


class InferenceHandler:
    question: Optional[str]
    context: Optional[str]

    def __init__(
        self,
        model_path: str,
    ):
        self._model_path = model_path
        self._context = None
        self._response = None
        self._question = None

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def model(self):
        return Llama(model_path=self._model_path)

    @property
    def question(self):
        return self._question

    @property
    def context(self):
        return self._context

    @property
    def response(self):
        return self._response

    @property
    def prompt(self) -> str:
        return f"""
            Use the following pieces of information to answer the user's question.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Context: {self.context}
            Question: {self.question}
            Only return the helpful answer below and nothing else.
            Helpful answer:
        """

    def wait_until_server_up(self) -> "InferenceHandler":
        while True:
            try:
                response = requests.get("http://client-server:5000/")
                if response.status_code == 200:
                    break
            except ConnectionError:
                time.sleep(3)
                continue
        return self

    def fetch_question(self) -> "InferenceHandler":
        self._question = requests.get("http://client-server:5000/question").json()
        return self

    def fetch_context(self) -> "InferenceHandler":
        response = requests.get("http://client-server:5000/documents")
        self._context = "\n".join(response.json())
        return self

    def generate(
            self,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_tokens=512,
    ) -> "InferenceHandler":
        output = self.model(
            self.prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens
        )
        self._response = output["choices"][0]["text"]
        return self.response

    def __call__(self, *args, **kwargs):
        return self.fetch_question().fetch_context().generate()

