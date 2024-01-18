from flask import Flask, jsonify, request
from inference import InferenceHandler
import requests

MODEL_PATH = "./models/"


app = Flask(__name__)

llm = InferenceHandler(MODEL_PATH)

@app.route("/", methods=["GET"])
def ping():
    return jsonify({'status': 'ok'})


@app.route("/inference", methods=["POST"])
def predict():
    input = request.get_json()
    question = input["question"]
    output = llm.set_question(question).wait_until_server_up().fetch_context().generate()
    payload = jsonify({"output": output})
    return payload


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port="5001")