import langchain
from langchain.document_loaders.dataframe import DataFrameLoader
from langchain.vectorstores import Chroma
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
import chromadb
from chromadb.config import Settings
import csv
import sys
import os
import argparse


class VectorDB:

    def __init__(
        self,
            data: pd.DataFrame
    ):
        self.data = data
        self.docs = None

    @property
    def embedding_model(self):
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

    @property
    def client(self):
        return chromadb.HttpClient(settings=Settings(allow_reset=True))

    def reset_db(self):
        self.client.reset()
        return self

    def load_documents(
        self, chunk_size: int = 500, chunk_overlap: int = 50
    ) -> None:
        csv.field_size_limit(sys.maxsize)
        loader = DataFrameLoader(self.data, page_content_column="documents")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.docs = text_splitter.split_documents(documents)
        return self

    def collections(self):
        self.load_documents()
        collection = self.client.reset_db().create_collection("mind_rag_collection")
        for doc in self.docs:
            collection.add(
                ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
            )
        return collection

    def set_up_vector_db(self) -> None:
        vectorstore = Chroma(
            client=self.client,
            collection_name="mind_rag_collection",
            embedding_function=self.embedding_model,
        )
        vectorstore = FAISS.from_documents(self.texts, self.embedding_model)
        vectorstore.save_local(self.db_path)
        print(f"Vector DB set up at {self.db_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path")
    args = parser.parse_args()
    data_path = os.path.dirname(os.path.abspath(__file__)) + "/data/nhs_mental_health_data.csv"
    data = pd.read_csv(data_path)
    if args.db_path:
        db = VectorDB(db_path=args.db_path)
    else:
        db = VectorDB()
    db.load_documents(data)
    db.set_up_vector_db()
