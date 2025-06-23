# Chat with PDF (OpenAI + Redis)

This project allows you to upload a PDF file and chat with it using AI. It uses OpenAI's GPT-4 to answer your questions based on the content of the uploaded PDF. It also uses Redis to store data and embeddings for fast and efficient responses.

---

## 💡 What It Does

- 📄 Upload any PDF
- 🤖 Ask questions about the content
- 🧠 AI understands and answers using GPT-4
- ⚡ Fast responses using Redis for memory and embeddings
- 🌐 Built with Streamlit for a simple web interface

---

## 🛠️ Tools & Libraries Used

- **Streamlit** – Web app interface  
- **PyPDF2** – To extract text from PDFs  
- **Sentence Transformers** – For creating vector embeddings  
- **Redis** – In-memory database to store text and embeddings  
- **LangChain + OpenAI** – For smart, context-aware responses  
- **GPT-4** – AI model used for answering questions  

---

## ⚙️ How It Works

### 1. Setup Environment
All libraries are imported and OpenAI API key is set in a secure way.

### 2. Redis Client
A Redis connection is made using host, port, and password. Connection is checked using `ping()`.

### 3. Load Embedding Model
Using `'all-MiniLM-L6-v2'` from Sentence Transformers. Cached for fast performance.

### 4. Memory for Chat
Stores last 5 messages using `ConversationBufferWindowMemory` to keep the conversation smooth.

### 5. LangChain Agent
GPT-4 is configured via LangChain with limited reasoning steps and strict instruction to only use PDF data.

### 6. Extract PDF Text
PDF is read using PyPDF2. All text is extracted and cleaned.

### 7. Store in Redis
Text and its embedding are saved in Redis — embedding in binary, text as string.

### 8. Find Most Relevant PDF
When user asks something, the query is embedded and compared with stored embeddings. Best match is selected.

### 9. Answer Generation
The selected text and question are passed to GPT-4 via LangChain to get an accurate response.

### 10. Streamlit Interface
Everything is shown in a clean Streamlit web app — from uploading to chatting, with helpful messages.

---


