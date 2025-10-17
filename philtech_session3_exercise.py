#### Import the libraries
import os
import uuid
from google import genai
from google.genai.types import HttpOptions, GenerateContentConfig
import vertexai
from qdrant_client import QdrantClient
from qdrant_client.http import models
from PyPDF2 import PdfReader
from vertexai.language_models import TextEmbeddingModel
import streamlit as st


# Specify Your GCP Project ID  
# Format Refer: Michael's project "gcp-abs-zz772-01-sbx-prj-01"
PROJECT_ID = "<ADD Your Code Here>"

# Create the VertexAI client
LLM_client = genai.Client(http_options=HttpOptions(api_version="v1"), project=PROJECT_ID, location="us-central1", vertexai=True)



#### Qdrant connection settings

# Specify your API_key and URL for Qdrant. You can get these from your Qdrant account https://login.cloud.qdrant.io/ 
# Format Refer: Michael's API_Key "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.u3SJa64k9qz6836q3y6pgr-55VNPftkkI3Ex0YZGyT8"
# Format Refer: Michael's URL "https://6c13d93e-46bd-425a-8f5f-fd7032d83d09.us-east4-0.gcp.cloud.qdrant.io:6333"
API_Key= "<ADD Your Code Here>"
URL = "<ADD Your Code Here>"



#### Qdrant functions

# Initialize the QdrantClient
def init_qdrant(qdrant_url: str, qdrant_api_key: str):
    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)


# Create a collection to host the uploaded files in Qdrant
def create_qdrant_collection(collection_name: str, qdrant_url: str, qdrant_api_key: str, 
                             vector_size: int = 3072, distance: str = "Cosine"):
    client = init_qdrant(qdrant_url, qdrant_api_key)
    try:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=distance)
        )
    except AttributeError:
        client.delete_collection(collection_name=collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=distance)
        )


# Ingest PDFs into Qdrant 
def ingest_pdfs_to_qdrant(pdf_files, collection_name: str, qdrant_url: str, qdrant_api_key: str, chunk_size: int = 500):
    client = init_qdrant(qdrant_url, qdrant_api_key)
    embedding_model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")

    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        full_text = "".join(page.extract_text() or "" for page in reader.pages)
        chunks = [full_text[i : i + chunk_size] for i in range(0, len(full_text), chunk_size)]
        embeddings = embedding_model.get_embeddings(chunks)

        points = []
        for emb, chunk in zip(embeddings, chunks):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb.values,
                    payload={"text": chunk}
                )
            )
        client.upsert(collection_name=collection_name, points=points)




#### Invoke LLM calls with user prompt and retrieved relevant context from the uploaded PDF
def get_bot_response(LLM_client, messages, model_name: str, temperature: float, top_p: float, max_output: int,
    collection_name: str, qdrant_url: str, qdrant_api_key: str, k: int = 3):
    """
    If `collection_name` is set, retrieve top-k chunks from Qdrant and prepend as context. 
    Otherwise, just chat on the conversation history.
    """
    # Build conversation history to allow for multi-turn conversation
    history = ""
    for msg in messages:
        speaker = "User" if msg["role"] == "user" else "Assistant"
        history += f"{speaker}: {msg['content']}\n"

    # Retrieve RAG context if available
    context = ""
    if collection_name:
        embedding_model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
        query_emb = embedding_model.get_embeddings([messages[-1]["content"]])[0].values
        client = init_qdrant(qdrant_url, qdrant_api_key)
        hits = client.search(collection_name=collection_name, query_vector=query_emb, limit=k)
        context = "\n\n".join(hit.payload.get("text", "") for hit in hits)

    # Compose prompt
    if context:
        prompt = f"Context:\n{context}\n\nConversation:\n{history}Assistant:"
    else:
        prompt = f"Conversation:\n{history}Assistant:"

    response = LLM_client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=GenerateContentConfig(
            temperature=temperature, top_p=top_p, max_output_tokens=max_output)
    )
    return response.text


#### Tweak and set LLM configurations


# Specify your LLM model - Note, input a string with quotation mark -> "model_name"
# Format Refer: gemini-2.5-flash, gemini-2.5-pro, gemini-2.5-flash-lite
MODEL_NAME = "<ADD Your Code Here>"


# Specify temperature - Note, input a decimal number
# Format Refer: 0.8
TEMPERATURE = "<ADD Your Code Here>"


# Specify Top_P - Note, input a decimal number
# Format Refer: 0.95
TOP_P = "<ADD Your Code Here>"


MAX_OUTPUT = 8192
DEFAULT_COLLECTION = "pdfs_collection"



#### UI


# Streamlit UI setup
st.set_page_config(page_title="My PDF RAG Chatbot", layout="wide", page_icon="🤖")
st.title("🤖 Chat with My PDFs")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qdrant_collection" not in st.session_state:
    st.session_state.qdrant_collection = None
    
# PDF uploader and automatic ingestion (only once) - reset after url refresh
uploaded_files = st.file_uploader(
    "📄 Upload PDFs (optional — chat works even without PDFs)",
    type=["pdf"],
    accept_multiple_files=True,
)
if uploaded_files and not st.session_state.qdrant_collection:
    with st.spinner("Creating Qdrant collection & ingesting PDFs…"):
        create_qdrant_collection(
            collection_name=DEFAULT_COLLECTION,
            qdrant_url=URL,
            qdrant_api_key=API_Key,
        )
        ingest_pdfs_to_qdrant(
            pdf_files=uploaded_files,
            collection_name=DEFAULT_COLLECTION,
            qdrant_url=URL,
            qdrant_api_key=API_Key,
        )
        st.session_state.qdrant_collection = DEFAULT_COLLECTION
        st.success("✅ Success!! PDFs ingested! Future replies will include RAG context.")
        
# Always-on chat interface: history above, input at bottom
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Type your question…"):
    # Append and display user turn
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    # Generate assistant reply (with or without RAG)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            reply = get_bot_response(
                LLM_client = LLM_client,
                messages=st.session_state.messages,
                model_name=MODEL_NAME,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_output=MAX_OUTPUT,
                collection_name=st.session_state.qdrant_collection,
                qdrant_url=URL,
                qdrant_api_key=API_Key,
            )
            st.markdown(reply)

    # Save assistant turn
    st.session_state.messages.append({"role": "assistant", "content": reply})

