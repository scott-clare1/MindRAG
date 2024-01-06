FROM python:3.8-slim-buster
RUN apt-get update && apt-get install -y netcat
WORKDIR /app
COPY wait-for-it.sh /app/wait-for-it.sh
RUN chmod +x /app/wait-for-it.sh
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8080
CMD ["./wait-for-it.sh", "localhost", "8000", "python", "app.py"]