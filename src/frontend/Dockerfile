FROM python:3.10-slim-buster
WORKDIR /app
COPY ./requirements /app
RUN pip install -r requirements.txt
COPY . /app
EXPOSE 5000
CMD ["streamlit", "run", "mind_rag_chat.py"]
