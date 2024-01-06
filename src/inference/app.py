from flask import Flask, jsonify, request
from inference import InferenceHandler


app = Flask(__name__)

llama_model = InferenceHandler("llama-2-7b-chat.Q4_K_M.gguf")


@app.route("/inference", methods=["POST"])
def predict():
    input = request.get_json()
    question = input["question"]
    output = llama_model.set_question(question).fetch_context().generate()
    payload = jsonify({"output": output})
    return payload


if __name__ == "__main__":
    app.run(debug=True)