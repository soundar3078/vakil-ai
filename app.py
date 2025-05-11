import time
import os
import logging
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline
from dotenv import load_dotenv
from PIL import Image
from footer import footer
from transformers import pipeline

# Streamlit UI Configuration
st.set_page_config(page_title="BharatLAW", layout="centered")

# Add the image banner at the top
try:
    image_path = os.path.join("images", "v.a.k.i.l.jpeg")
    image = Image.open(image_path)
    st.image(image, use_column_width=True)
except Exception as e:
    st.warning(f"Could not load banner image: {e}")
    st.image("https://github.com/Nike-one/BharatLAW/blob/master/images/banner.png?raw=true", use_column_width=True)

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bharatlaw.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

# Hide default Streamlit UI elements
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stChatInput {bottom: 20px;}
    </style>
""", unsafe_allow_html=True)

# Hugging Face Key Setup
hf_api_key = os.getenv('HUGGINGFACE_API_TOKEN')
if not hf_api_key:
    st.error("""
        ❌ HUGGINGFACE_API_TOKEN not found in .env file.
        Please add your key to .env file like:
        HUGGINGFACE_API_TOKEN=your_token_here
    """)
    st.stop()

# Initialize the Flan-T5-large model using transformers pipeline and LangChain wrapper
@st.cache_resource
def load_llm():
    base_pipeline = pipeline('text2text-generation', model='google/flan-t5-large', device_map='auto',
                            model_kwargs={"temperature": 0.5, "max_length": 512})
    llm = HuggingFacePipeline(pipeline=base_pipeline)
    return llm

try:
    llm = load_llm()
    logging.info("HuggingFace Flan-T5-large model initialized successfully")
except Exception as e:
    st.error(f"LLM Initialization Failed: {str(e)}")
    st.stop()

# Load Embeddings
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")

try:
    embeddings = load_embeddings()
    db = FAISS.load_local("ipc_embed_db", embeddings, allow_dangerous_deserialization=True)
    db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
except Exception as e:
    st.error(f"Vector Database Error: {str(e)}")
    st.stop()

# Initialize chat history and memory
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)

# Prompt Template
prompt_template = PromptTemplate(
    input_variables=["question", "context", "chat_history"],
    template="""
You are VAKIL AI, an expert legal assistant specializing in the Indian Penal Code (IPC).

INSTRUCTIONS:
- Use the provided context to answer the user's question accurately and concisely.
- Do not provide information outside of the context.
- If the context does not contain the answer, respond with "I don't know."
- Format your answer as a direct response to the question.

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

QUESTION:
{question}

ANSWER:
"""
)

# QA chain with ConversationalRetrievalChain
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db_retriever,
    memory=st.session_state.memory,
    combine_docs_chain_kwargs={"prompt": prompt_template}
)

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input and response generation
if prompt := st.chat_input("Ask your legal question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        try:
            # Retrieve relevant documents
            docs = db_retriever.get_relevant_documents(prompt)

            # Log retrieved documents - ENHANCED LOGGING
            logging.info("Retrieved Documents:")
            for i, doc in enumerate(docs):
                logging.info(f"  --- Document {i + 1} ---")
                logging.info(f"  Content: {doc.page_content}")
                logging.info(f"  Metadata: {doc.metadata}")  # Log metadata if available

            # Display retrieved documents in Streamlit (for debugging)
            with st.expander("Retrieved Context (DEBUG)"):
                for doc in docs:
                    st.write(doc.page_content)
                    st.write("---")  # Separator for clarity

            context = "\n".join([doc.page_content for doc in docs])

            # Format chat history for the prompt
            chat_history_str = "\n".join(
                [f"{m['role']}: {m['content']}" for m in st.session_state.messages if m["role"] != "system"])

            # Prepare input for the QA chain
            qa_input = {"question": prompt, "chat_history": chat_history_str}

            # Log the input to the LLM
            logging.info(f"LLM Input: {qa_input}")

            # Generate the response using the QA chain
            response = qa.invoke(qa_input)
            full_response = response["answer"]

            message_placeholder = st.empty()
            display_text = ""
            for chunk in full_response.split():
                display_text += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(display_text + "▌")
            message_placeholder.markdown(display_text)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            logging.error(f"Generation Error: {str(e)}")

footer()

# Optional: Add a button to clear chat history in the sidebar
if st.sidebar.button("Clear Chat History"):
    st.session_state.memory.clear()
    st.session_state.messages = []
    st.rerun()