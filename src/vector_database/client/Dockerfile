FROM python:3.8-slim-buster
WORKDIR /app
COPY ./src/vector_database/client/requirements.txt /app
RUN pip install -r /app/requirements.txt
COPY ./src/vector_database/client .
EXPOSE 5000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000", "--reload"]