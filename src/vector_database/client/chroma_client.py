import pysqlite3
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import pandas as pd
import uuid
import chromadb
from chromadb.config import Settings
import csv
import sys
from langchain.document_loaders.dataframe import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from requests.exceptions import ConnectionError
import time


class ChromaClient:

    def __init__(
            self,
            data: pd.DataFrame
    ):
        self.data = data
        self.docs = None
        self.collection = None
        self._context_documents = None
        self._context_urls = None
        self._context_titles = None
        self._question = None
        self.collection_name = "mind_rag_collection"

    @property
    def client(self):
        return chromadb.HttpClient(host="vector-db", port="8000", settings=Settings(allow_reset=True))

    @property
    def context_documents(self):
        return self._context_documents

    @property
    def context_urls(self):
        return self._context_urls

    @property
    def context_titles(self):
        return self._context_titles

    @property
    def question(self):
        return self._question

    def wait_until_server_up(self) -> "VectorDB":
        while True:
            try:
                response = requests.get("http://vector-db:8000/api/v1/heartbeat")
                if response.status_code == 200:
                    break
            except ConnectionError:
                time.sleep(3)
                continue
        return self

    def reset_db(self) -> "VectorDB":
        self.client.reset()
        return self

    def build_documents(
            self, chunk_size: int = 300, chunk_overlap: int = 50
    ) -> "VectorDB":
        csv.field_size_limit(sys.maxsize)
        loader = DataFrameLoader(self.data, page_content_column="documents")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.docs = text_splitter.split_documents(documents)
        return self

    def build_collections(self) -> "VectorDB":
        self.collection = self.client.get_or_create_collection(self.collection_name)
        for doc in self.docs:
            self.collection.add(
                ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
            )
        return self

    def query(self, question: str) -> "VectorDB":
        self._question = question
        response = self.collection.query(
            query_texts=[self.question],
            n_results=10,
        )
        self._context_documents = response["documents"][0]

        metadata = response["metadatas"][0]
        self._context_urls = [item["links"] for item in metadata]
        self._context_titles = [item["title"] for item in metadata]
        return self
