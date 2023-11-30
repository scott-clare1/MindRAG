from flask import Flask, jsonify, request
from inference import VectorDBQuery


app = Flask(__name__)

vector_db_query = VectorDBQuery("llama-2-7b-chat.Q4_K_M.gguf")


@app.route("/inference", methods=["POST"])
def predict():
    input = request.get_json()
    question = input["question"]
    output = vector_db_query(question)
    payload = jsonify({"output": output})
    return payload


if __name__ == "__main__":
    app.run(debug=True)