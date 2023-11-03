import os

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.text_splitter import CharacterTextSplitter
#from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS

#from frontend import css, bot_template, user_template

# Load environment variables from .env file (Optional)
load_dotenv()

OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")

system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

def main():
    # Set the title and subtitle of the app
    st.title('🦜DLS Assistance Website Chat🔗')
    st.subheader("Developed By Natnael M.")
    
    with st.sidebar.expander("Settings", expanded=True):
        MODEL = st.selectbox(label='Model', options=['gpt-4','gpt-3.5-turbo','text-davinci-003','text-davinci-002','code-davinci-002'])
        PERSONALITY = st.selectbox(label='Personality', options=['general assistant','academic'])
        TEMP = st.slider("Temperature",0.0,1.0,0.5)
        st.subheader('Input your website URL, ask questions, and receive answers directly from the website.')

    url = st.text_input("Insert The website URL")

    prompt = st.text_input("Write Your Question Here..")
    if st.button("Submit", type="primary"):
        ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
        DB_DIR: str = os.path.join(ABS_PATH, "db")

        # Load data from the specified URL
        loader = WebBaseLoader(url)
        data = loader.load()

        # Split the loaded data
        text_splitter = CharacterTextSplitter(separator='\n', 
                                        chunk_size=500, 
                                        chunk_overlap=40)

        docs = text_splitter.split_documents(data)

        # Create OpenAI embeddings
        openai_embeddings = OpenAIEmbeddings()

        # vectordb.persist()
        vectordb = FAISS.from_documents(
            documents=docs,
            embedding=openai_embeddings,
            #persist_directory=DB_DIR

        )

        # Create a retriever from the Chroma vector database
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        # Use a ChatOpenAI model
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        #MODEL = st.selectbox(label='Model', options=['gpt-3.5-turbo','text-davinci-003','text-davinci-002','code-davinci-002'])

        # Create a RetrievalQA from the model and retriever
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        # Run the prompt and return the response
        response = qa(prompt)
        st.write(response)

# if __name__ == '__main__':
#     main()
if __name__ == '__main__':
    load_dotenv()
    main()
