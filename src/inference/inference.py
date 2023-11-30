from langchain.llms import CTransformers
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
import langchain


EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
)


class VectorDBQuery:
    def __init__(
        self,
        model_path: str,
        embedding_model: langchain.embeddings.huggingface.HuggingFaceEmbeddings = EMBEDDING_MODEL,
        temperature: float = 0.01,
        max_new_tokens: int = 300,
    ):
        self.llm = CTransformers(
            model=model_path,
            model_type="llama",
            config={"max_new_tokens": max_new_tokens, "temperature": temperature},
        )
        self.qa_template = """Use the following pieces of information to answer the user's question.
                            If you don't know the answer, just say that you don't know, don't try to make up an answer.
                            Context: {context}
                            Question: {question}
                            Only return the helpful answer below and nothing else.
                            Helpful answer:
                            """
        self.embedding_model = embedding_model

        self.vectordb = FAISS.load_local("vectorstore/db_faiss", self.embedding_model)
        self.prompt = PromptTemplate(
            template=self.qa_template, input_variables=["context", "question"]
        )
        self.dbqa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectordb.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt},
        )

    def __call__(self, input: str) -> str:
        response = self.dbqa(input)
        return response
