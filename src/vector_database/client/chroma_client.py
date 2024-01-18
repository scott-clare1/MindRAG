import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import pandas as pd
import uuid
import chromadb
from chromadb.config import Settings
import csv
import sys
import os
import argparse
from langchain.document_loaders.dataframe import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from requests.exceptions import ConnectionError
import time


class VectorDB:

    def __init__(
        self,
            data: pd.DataFrame
    ):
        self.data = data
        self.docs = None
        self.collection = None
        self.collection_name = "mind_rag_collection"

    def wait_until_server_up(self):
        while True:
            try:
                response = requests.get("http://vector-db:8000/api/v1/heartbeat")
                if response.status_code == 200:
                    break
            except ConnectionError:
                time.sleep(3)
                continue
        return self

    @property
    def client(self):
        return chromadb.HttpClient(host="vector-db", port="8000", settings=Settings(allow_reset=True))

    def reset_db(self):
        self.client.reset()
        return self

    def build_documents(
        self, chunk_size: int = 500, chunk_overlap: int = 50
    ):
        csv.field_size_limit(sys.maxsize)
        loader = DataFrameLoader(self.data, page_content_column="documents")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.docs = text_splitter.split_documents(documents)
        return self

    def build_collections(self):
        self.collection = self.client.get_or_create_collection(self.collection_name)
        for doc in self.docs:
            self.collection.add(
                ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
            )
        return self

    def query(self, question: str):
        response = self.collection.query(
            query_texts=[question],
            n_results=10,
        )
        documents = response["documents"][0]
        metadata = response["metadatas"][0]
        links = [item["links"] for item in metadata]
        titles = [item["title"] for item in metadata]
        return {
            "documents": documents,
            "links": links,
            "title": titles
        }
