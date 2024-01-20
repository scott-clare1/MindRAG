import streamlit as st
import requests
from typing import Dict, List


def build_output_text(response: dict) -> str:
    text = """
    Unfortunately, I was not able to provide a coherent response to this query - 
    could you try asking again and making sure the question is about mental health?
    """
    result = response["result"]
    if len(set(result.split())) > len(result.split()) // 2:
        statement = []
        result_html = "<p>" + result + "</p>"
        statement.append(result_html)
        statement.append("<p>Generated from the following documents:</p>")
        source_documents = response["source_documents"]
        for doc in source_documents:
            content_html = f'<a href="{doc.metadata["links"]}">{doc.metadata["title"].strip()}</a><br>'
            if content_html not in statement:
                statement.append(content_html)
        text = f"{' '.join(statement)}"
    return text


def create(question: str):
    return requests.post("http://client-server:5000/query", json={"question": question})


def get_documents():
    return requests.get("http://client-server:5000/documents").json()


def get_urls():
    return requests.get("http://client-server:5000/urls").json()


def get_titles():
    return requests.get("http://client-server:5000/titles").json()


def get_answer():
    return requests.get("http://llm-inference:5001/inference").json()



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
                create(i["content"])
                output = get_response()
                st.write(output, unsafe_allow_html=True)
                st.session_state["chat_history"][idx]["content"] = output
                status.update(label="Complete!", state="complete", expanded=True)
        else:
            st.write(i["content"], unsafe_allow_html=True)
