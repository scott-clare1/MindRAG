from flask import Flask, jsonify, request
from chroma_client import VectorDB
import pandas as pd

app = Flask(__name__)

data = pd.read_csv("data/nhs_mental_health_data.csv")

vector_db_client = VectorDB(data).wait_until_server_up().build_documents().build_collections()

@app.route("/", methods=["GET"])
def ping():
    return jsonify({'status': 'ok'})

@app.route("/query", methods=["POST"])
def query():
    input = request.get_json()
    question = input["question"]
    output = vector_db_client.query(question)
    payload = jsonify({"output": output})
    return payload

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)