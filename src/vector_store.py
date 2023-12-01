import langchain
from langchain.document_loaders.dataframe import DataFrameLoader
from langchain.vectorstores import FAISS
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import csv
import sys
from inference.inference import EMBEDDING_MODEL


class VectorDB:
    def __init__(
        self,
        embedding_model: langchain.embeddings.huggingface.HuggingFaceEmbeddings = EMBEDDING_MODEL,
    ):
        self.embedding_model = embedding_model
        self.db_path = "vectorstore/db_faiss"
        self.texts = None

    def load_documents(
        self, data: pd.DataFrame, chunk_size: int = 500, chunk_overlap: int = 50
    ) -> None:
        csv.field_size_limit(sys.maxsize)
        loader = DataFrameLoader(data, page_content_column="documents")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.texts = text_splitter.split_documents(documents)

    def set_up_vector_db(self) -> None:
        vectorstore = FAISS.from_documents(self.texts, self.embedding_model)
        vectorstore.save_local(self.db_path)
        print(f"Vector DB set up at {self.db_path}")


if __name__ == "__main__":
    data = pd.read_csv("data/nhs_mental_health_data.csv")
    db = VectorDB()
    db.load_documents(data)
    db.set_up_vector_db()
