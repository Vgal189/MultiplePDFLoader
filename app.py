import os
import pickle
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
from langchain.callbacks import get_openai_callback
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI

load_dotenv()

loader = DirectoryLoader('./files/', glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
texts = text_splitter.split_documents(documents)

store_name = "localSave"

if os.path.exists(f"localSave.pkl"):
    with open(f"localSave.pkl", "rb") as f:
        vector_store = pickle.load(f)
    print("Loaded from disk")
else:
    with get_openai_callback() as cb:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(documents, embedding=embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(vector_store, f)
        print(cb)
        print('Embeddings costed')

query = input("Ask your question:")

if query:
    retriever = vector_store.as_retriever()
    docs = retriever.get_relevant_documents(query)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    llm = ChatOpenAI(temperature=0.5, model_name='gpt-3.5-turbo')
    chain = RetrievalQA.from_chain_type(llm=llm, 
                                  chain_type="stuff", 
                                  retriever=retriever) 
    with get_openai_callback() as cb:
        response = chain.run(query)
        print(cb)
    print(response)
