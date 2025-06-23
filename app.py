import os
import redis
import streamlit as st
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.agents import initialize_agent, AgentType
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI

# ------------------ Streamlit UI Setup ------------------
st.set_page_config(page_title="PDF Chatbot", layout="wide")

# ------------------ Environment Setup ------------------
os.environ["OPENAI_API_KEY"] = ''  # Add your OpenAI API key here

# ------------------ Redis Client Setup ------------------
r = redis.Redis(host="", port="", password="")  # Fill with your actual Redis config

# Verify Redis connection
try:
    r.ping()
    st.success("Connected to Redis successfully!")
except redis.ConnectionError as e:
    st.error(f"Could not connect to Redis: {e}")

# ------------------ Load Sentence Transformer ------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ------------------ Conversational Memory ------------------
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

# ------------------ System Prompt ------------------
system_prompt = """
You are an advanced assistant that has access to the content of uploaded PDF documents.
Your goal is to provide exact answers directly from the PDF based on the user's query. 
Make sure to only refer to the text within the document and give clear, concise, and accurate answers. 
"""

# ------------------ OpenAI LLM Setup ------------------
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.2,
)

agent = initialize_agent(
    tools=[],
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    llm=llm,
    max_iterations=5,
    verbose=True,
    memory=conversational_memory,
    early_stopping_method='generate'
)

# ------------------ PDF Processing Functions ------------------

def extract_pdf_text(uploaded_file):
    """Extract text from a PDF file."""
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    if not text:
        st.warning("No readable text found in the uploaded PDF.")
    return text

def store_embeddings_in_redis(pdf_text, pdf_name):
    """Store PDF text and embeddings in Redis."""
    pdf_key = pdf_name.replace(" ", "_")
    embedding = model.encode([pdf_text])[0].astype(np.float32)

    # Store embedding and PDF text separately
    r.set(f"{pdf_key}_embedding", embedding.tobytes())
    r.set(f"{pdf_key}_text", pdf_text)

def retrieve_similar_pdf(query):
    """Retrieve the most relevant PDF based on query embedding similarity."""
    query_embedding = model.encode([query])[0].astype(np.float32)
    similarities = []

    for key in r.keys("*_embedding"):
        key_str = key.decode('utf-8').replace("_embedding", "")
        stored_embedding = np.frombuffer(r.get(key), dtype=np.float32)
        sim = cosine_similarity([query_embedding], [stored_embedding])[0][0]
        similarities.append((key_str, sim))

    if similarities:
        similarities.sort(key=lambda x: x[1], reverse=True)
        best_match_key = similarities[0][0]
        best_text = r.get(f"{best_match_key}_text").decode("utf-8")
        return best_match_key, best_text
    return None, None

def get_answer_from_openai(query, context):
    """Generate answer using OpenAI with PDF context."""
    prompt = f"""
Context from the PDF:
{context}

User query:
{query}

Based only on the above context, please answer the question clearly.
"""
    return agent.run(prompt)

# ------------------ Streamlit App ------------------

st.title("📄 Chat with Your PDF (OpenAI + Redis)")

pdf_file = st.file_uploader("Upload a PDF", type="pdf")

if pdf_file:
    with st.spinner("Extracting and storing embeddings..."):
        pdf_text = extract_pdf_text(pdf_file)
        if pdf_text:
            store_embeddings_in_redis(pdf_text, pdf_file.name)
            st.success(f"✅ Stored embeddings and text for {pdf_file.name}")

user_query = st.text_input("💬 Ask something from your PDFs:")

if user_query:
    with st.spinner("Retrieving relevant documents..."):
        matched_pdf, matched_text = retrieve_similar_pdf(user_query)
        if matched_pdf:
            st.write(f"📚 Most relevant document: **{matched_pdf}**")
            response = get_answer_from_openai(user_query, matched_text)
            st.write(f"🤖 Response: {response}")
        else:
            st.warning("No documents found in Redis to match your query.")
