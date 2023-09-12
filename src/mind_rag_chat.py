import streamlit as st
from rag import VectorDBQuery, embedding_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--llm_path", action="store", type=str)
parser.add_argument("--temperature", action="store", type=float, default=0.01)
parser.add_argument("--max_new_tokens", action="store", type=int, default=300)
try:
    args = parser.parse_args()
except SystemExit as e:
    os._exit(e.code)


@st.cache_resource
def load_query_agent():
    return VectorDBQuery(
        embedding_model, args.llm_path, args.temperature, args.max_new_tokens
    )


response = load_query_agent()

st.title("MindRAG")
st.write(
    "My knowledge is based on documents from the NHS's many pages on mental health conditions - feel free to ask a question related to mental health conditions!"
)


def chat_actions():
    st.session_state["chat_history"].append(
        {"role": "user", "content": st.session_state["chat_input"]},
    )
    st.session_state["chat_history"].append(
        {
            "role": "assistant",
            "content": st.session_state["chat_input"],
        }
    )


if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


st.chat_input(
    "Ask any mental health related questions here...",
    on_submit=chat_actions,
    key="chat_input",
)

for idx, i in enumerate(st.session_state["chat_history"]):
    with st.chat_message(name=i["role"]):
        if i["role"] == "user":
            st.write(i["content"])
        elif idx == len(st.session_state["chat_history"]) - 1:
            with st.status("Thinking...", expanded=True) as status:
                output = response(i["content"])
                st.write(output, unsafe_allow_html=True)
                st.session_state["chat_history"][idx]["content"] = output
                status.update(label="Complete!", state="complete", expanded=True)
        else:
            st.write(i["content"], unsafe_allow_html=True)
