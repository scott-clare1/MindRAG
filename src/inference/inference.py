import requests
from typing import Optional
from typing import Dict, List, Union


class LLMClient:
    _question: Optional[str] = None
    _context: Optional[str] = None
    _response: Optional[Dict[str, Union[str, int, List[dict]]]] = None

    @property
    def question(self) -> Optional[str]:
        return self._question

    @property
    def context(self) -> Optional[str]:
        return self._context

    @property
    def response(self) -> Optional[Dict[str, Union[str, int, List[dict]]]]:
        return self._response

    @property
    def answer(self) -> Optional[str]:
        if self.response:
            return self.response["choices"][0]["text"]
        else:
            return None

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

    @property
    def payload(self) -> Dict[str, Union[str, List]]:
        return {
            "prompt": self.prompt,
            "stop": ["\n", "###"]
        }

    def fetch_question(self) -> "LLMClient":
        self._question = requests.get("http://client-server:5000/question").json()
        return self

    def fetch_context(self) -> "LLMClient":
        response = requests.get("http://client-server:5000/documents")
        self._context = "\n".join(response.json())
        return self

    def fetch_answer(self) -> str:
        self._response = requests.post("http://llama-server:8000/v1/completions", json=self.payload).json()
        return self.answer

    def __call__(self) -> str:
        return self.fetch_question().fetch_context().fetch_answer()

