#   V.A.K.I.L AI: AI IPC Legal advice Assistant 📘

BharatLAW is a sophisticated legal advisory chatbot focused on providing detailed and contextually accurate responses about the Indian Penal Code. It utilizes a powerful combination of machine learning technologies to efficiently process and retrieve legal information.

---

##   Features 🌟

* **Document Ingestion**:   Automated processing of text documents to store legal information in a FAISS vector database.
* **Real-Time Interaction**:   Real-time legal advice through a conversational interface built with Streamlit.
* **Legal Prompt Templating**:   Structured prompt format ensuring clarity, detail, and legal accuracy in responses.

<br>

---

<h4><strong>🚀Blast off to discovery! Our project is waiting for you <a href= "https://huggingface.co/spaces/nik-one/BharatLAW-IPC_legal_guidance">BharatLAW</a>. Explore it today and elevate your understanding!🌟</strong><h4>

<br>

---

##   Components 🛠️

###   Ingestion Script (`Ingest.py`)

|   Functionality        |   Description                                                                  |
| :--------------------- | :--------------------------------------------------------------------------- |
|   **Document Loading** |   Loads text documents from a specified directory.                             |
|   **Text Splitting** |   Splits documents into manageable chunks for processing.                      |
|   **Embedding Generation** |   Utilizes `HuggingFace's InLegalBERT` to generate text embeddings.           |
|   **FAISS Database** |   Indexes embeddings for fast and efficient retrieval.                        |

###   Web Application (`app.py`)

|   Feature               |   Description                                                                                                |
| :-------------------- | :--------------------------------------------------------------------------------------------------------- |
|   **Streamlit Interface** |   Provides a web interface for user interaction.                                                             |
|   **Chat Functionality** |   Manages conversational flow and stores chat history.                                                      |
|   **Legal Information Retrieval** |   Leverages FAISS index to fetch pertinent legal information based on queries.                           |
|   **Hugging Face API** |   Integrates with the Hugging Face Inference API to access and query language models for generating responses. |

---

##   Setup 📦

###   Prerequisites

* Python 3.8 or later
* ray
* langchain
* streamlit
* faiss
* requests  # Added

###   Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/your-repo/BharatLAW.git](https://github.com/your-repo/BharatLAW.git)
    cd BharatLAW
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up the Hugging Face API Token:**

    Obtain an API token from <a href="https://huggingface.co/">Hugging Face</a>.
    You'll need this to authenticate with the Hugging Face Inference API.

    **>   To set this API token as an environment variable, you can use the following approach:**

    * On macOS and Linux:

        ```bash
        echo "export HUGGINGFACE_API_TOKEN='Your-HuggingFace-API-Token-Here'" >> ~/.bash_profile
        source ~/.bash_profile
        ```

    * On Windows (using Command Prompt):

        ```cmd
        setx HUGGINGFACE_API_TOKEN "Your-HuggingFace-API-Token-Here"
        ```

    * On Windows (using PowerShell):

        ```powershell
        [Environment]::SetEnvironmentVariable("HUGGINGFACE_API_TOKEN", "Your-HuggingFace-API-Token-Here", "User")
        ```

    This token is crucial for the chatbot to access language model functionalities provided by Hugging Face.

##   Running the Application

1.  **Run the ingestion script to prepare the data:**

    ```bash
    python ingest.py
    ```

2.  Launch the Streamlit application:

    ```bash
    streamlit run app.py
    ```

---

##   Usage 🔍

Navigate to the local URL provided by Streamlit to interact with the BharatLAW chatbot. Enter your legal queries and receive precise information derived from the indexed legal documents. Utilize the chat interface to engage in a legal discussion and get accurate advice."# vakil-ai" 
