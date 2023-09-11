# MindRAG #

MindRAG is an application to ask natural language questions regarding mental health - the application uses retrieval augemented generation to answer users questions based on a knowledge base on 
mental health taken from the following resource https://www.nhs.uk/mental-health/conditions/.

The application is set up to be run locally and so requires the user to download a llama model for text generation. I have used a quantized llama-2-7b model (llama-2-7b-chat.ggmlv3.q2_K.bin) which 
can be downloaded from HuggingFace here: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML. This model uses the most aggresive quantization possible reducing the floating-point precision down to 
2-bit - this has meant that the application can return responses in roughly 10 seconds on a Macbook Pro.

The application is very simple and relies on existing techologies like langhcain, sentence-transformers and of course llama-2. First we create a vector database of the scraped NHS data using a 
sentence-transformers embedding model - this model creates highly dimensional vector representations or embeddings of 500 token chunks of the data. Once the vector db is set up we can retrieve 
semantically similar documents based on a prompt from the streamlit app e.g. what is depression? This prompt is also passed through the embedding model which is used as query to the db to find 
similar documents. These similar documents are then passed into a question-answer template with the retrieved documents used as context. The question-answer template can be seen below:

```
"""
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""
```

In this instance "question" is the prompt "what is depression?" and the context will hopefully be whatever nhs documents that discuss depression.

This question-answer prompt is then used to be passed into our large language model to provide a generated response based on the context given.

Finally, because this is a retrieval augmented task we can return the source texts with links for further reading for the user.


## Installation ## 
The requirements.txt file contains all the dependencies needed to run the app. Install these in whichever virtual environment you are using as below:
```
pip install -r requirments.txt
```

## Setting up Vector Database ##
The vector database can be set up with the following line of code:

```
python -m vector_store.py
```

## Running Streamlit App ##
As discussed previously you will need to download whichever llama model you wish to use and then save wherever you like - in the below example I have saved the model in the models subdirectory.
```
streamlit run src/mind_rag_chat.py -- --llm_path models/llama-2-7b-chat.ggmlv3.q2_K.bin --temperature 0.01 --max_new_tokens 300
```

## Application in action ##

