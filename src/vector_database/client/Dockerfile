FROM python:3.8-slim-buster
WORKDIR /app
COPY ./src/vector_database/client/requirements.txt /app
RUN pip install -r /app/requirements.txt
COPY ./src/vector_database/client .
EXPOSE 5000
ENV FLASK_APP=app.py
CMD ["flask", "run"]