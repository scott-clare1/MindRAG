from transformers import AutoModelForCausalLM, AutoTokenizer
import requests


class InferenceHandler:
    def __init__(
        self,
        model_path: str,
    ):
        self._model_path = model_path
        self.question = None
        self.context = None
        self._model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    def set_question(self, question: str):
        self.question = question
        return self

    def fetch_context(self):
        response = requests.post("http://127.0.0.1:5000/query", json={"question": self.question})
        self.context = "\n".join(response.json()["output"])
        return self

    @property
    def prompt(self):
        return f"""Use the following pieces of information to answer the user's question.
                            If you don't know the answer, just say that you don't know, don't try to make up an answer.
                            Context: {self.context}
                            Question: {self.question}
                            Only return the helpful answer below and nothing else.
                            Helpful answer:
                            """

    def generate(
            self,
            use_cuda = False,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_new_tokens=512,
            do_sample=True
    ):
        if use_cuda:
            input_ids = tokenizer(self.prompt, return_tensors='pt').input_ids.cuda()
        else:
            input_ids = tokenizer(self.prompt, return_tensors='pt').input_ids
        output = self.model.generate(
            inputs=input_ids,
            temperature=temperature,
            do_sample=do_sample, top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens
        )
        return tokenizer.decode(output[0])
