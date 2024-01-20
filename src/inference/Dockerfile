FROM python:3.10-slim-buster
WORKDIR /app
COPY ./src/inference/requirements.txt /app
ENV CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
RUN pip install -r requirements.txt
COPY ./src/inference /app
EXPOSE 5001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000", "--reload"]
