FROM python:3.10-slim-buster
WORKDIR /app
COPY ./src/frontend/requirements.txt /app
RUN pip install -r /app/requirements.txt
COPY ./src/frontend /app
EXPOSE 8080
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
