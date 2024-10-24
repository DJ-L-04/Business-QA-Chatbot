import openai
import langchain
import pinecone
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from pinecone import ServerlessSpec
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
st.set_page_config(
    page_title="Yardstick Chat Assistant",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
    <style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
        background-color: #f5f7fa;
    }
        
    .chat-message {
        padding: 0.75rem;
        border-radius: 15px;
        margin-bottom: 0.75rem;
        display: inline-block;
        max-width: 70%;
        min-width: 20%;
        line-height: 1.3;
    }
    
    .chat-message > div:first-child {
        margin-bottom: 0.25rem;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    .bot-message {
        background-color: #e8eef9;
        border-bottom-left-radius: 5px;
        float: left;
        clear: both;
        color: #2c3e50;
    }
    
    .user-message {
        background-color: #3498db;
        color: white;
        border-bottom-right-radius: 5px;
        float: right;
        clear: both;
    }
    
    .input-container {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 1rem;
        background-color: white;
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 100;
        border-top: 1px solid #e0e3e9;
    }
    
    .stTextInput {
        padding-bottom: 0 !important;
    }
    
    .block-container {
        padding-bottom: 5rem;
        background-color: #f5f7fa;
    }
    
    div[data-testid="stVerticalBlock"] > div:has(.stTextInput) {
        display: flex;
        gap: 0.5rem;
    }
    
    button {
        border-radius: 15px !important;
        padding: 0 1.5rem !important;
        background-color: #3498db !important;
    }
    
    button:hover {
        background-color: #2980b9 !important;
    }
    
    .stTextInput > div {
        padding-bottom: 0 !important;
    }
    
    .stTextInput input {
        border-radius: 15px;
        border: 1px solid #e0e3e9;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Title styling */
    h1 {
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

#Read document
@st.cache_resource
def read_doc(directory):
    loader = PyPDFDirectoryLoader(directory)
    documents = loader.load()
    return documents

#Text chunks
@st.cache_resource
def text_chunks(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    return chunks

#Embeddings
@st.cache_resource
def initialize_embeddings_and_index():

    os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    
    embeddings = OpenAIEmbeddings(api_key= OPENAI_API_KEY)
    
    from pinecone import Pinecone
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    
    if 'my-index' not in pc.list_indexes().names():
        pc.create_index(
            name='my-index',
            dimension=1536,
            metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    
    documents = read_doc('Yardstick.pdf')
    chunks = text_chunks(documents)
    from langchain.vectorstores import Pinecone
    index = Pinecone.from_documents(chunks, embeddings, index_name='my-index')
    
    return index

# Cosine similarity for results from vectorDB
def retrieve_query(query, k=2):
    index= initialize_embeddings_and_index()
    matching_results = index.similarity_search(query, k=k)
    return matching_results

# Initialize LLM and chain
llm = OpenAI(model_name='text-davinci-003', temperature=0.5)
chain = load_qa_chain(llm, chain_type='stuff')

def retrieve_answers(query, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            answer = retrieve_query(query)
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=query,
                max_tokens=100
            )
            return response.choices[0].text.strip()
        except Exception as e:
            break
    return "Unable to retrieve answer due to API limits."

def main():
    st.title("🤖 Yardstick QA Chatbot")
    st.markdown("<h4 style='text-align:center';>Lexy : Your smart companion for seamless conversations</h4>",unsafe_allow_html=True)
    st.markdown("---")

    with st.spinner("Initializing the chat system..."):
        index = initialize_embeddings_and_index()

    chat_container = st.container()
    
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                    <div class="chat-message user-message">
                        <div>You</div>
                        <div>{message["content"]}</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message bot-message">
                        <div>Lexy</div>
                        <div>{message["content"]}</div>
                    </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        cols = st.columns([3, 1, 1])
        with cols[0]:
            user_question = st.text_input("", placeholder="Type your question here", 
                                        label_visibility="collapsed", key="user_input")
        with cols[1]:
            ask_button = st.button("Ask Lexy")
        with cols[2]:
            clear_button = st.button("Clear Chat")

    if ask_button and user_question:
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })

        with st.spinner("Thinking..."):
            response = retrieve_answers(user_question)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })

    if clear_button:
        st.session_state.chat_history = []

if __name__ == "__main__":
    main()