
# 📘 PDF-Based RAG Question Answering using LangChain (Latest Version)

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG)** system using:

* **LangChain v0.2+**
* **OpenAI GPT Models**
* **FAISS vector store**
* **Embeddings for semantic search**
* **A PDF as the knowledge base**

The app loads a PDF, splits it into chunks, embeds it, performs semantic retrieval, and finally answers questions using only the relevant context extracted from the PDF.

---

## ✨ Features

* 📄 **PDF ingestion**
* 🔍 **Text chunking with overlap**
* 🧠 **OpenAI embeddings**
* 📚 **FAISS semantic vector search**
* 🤖 **GPT model for answering questions**
* 🔗 **Modern LangChain chain architecture**
* 🛠️ **Fully updated for 2024–2025 LangChain API**

---

## 🚀 Getting Started

### 1. **Clone the project**

```bash
git clone https://github.com/your-username/pdf-rag-langchain.git
cd pdf-rag-langchain
```

---

## ⚙️ Installation

Install the dependencies:

```bash
pip install -r requirements.txt
```

A typical `requirements.txt`:

```
langchain
langchain-openai
langchain-community
langchain-text-splitters
faiss-cpu
python-dotenv
pypdf
```

---

## 🔑 Environment Setup

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key_here
```

Load it inside the script:

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## 📄 Input PDF

Place your document in the project folder, for example:

```
attention.pdf
```

---

# 🧩 Full Application Code

```python
# ---------------------------
# 1. PDF Loader + Text Splitter
# ---------------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

loader = PyPDFLoader("attention.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunk_docs = splitter.split_documents(docs)


# ---------------------------
# 2. Embeddings + FAISS Vector Store
# ---------------------------
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunk_docs, embeddings)

retriever = vectorstore.as_retriever()


# ---------------------------
# 3. LLM
# ---------------------------
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# ---------------------------
# 4. Prompt Template
# ---------------------------
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the question ONLY using the provided context.\n"
     "Think step by step before answering.\n\n"
     "<context>\n{context}\n</context>"
    ),
    ("user", "{input}")
])


# ---------------------------
# 5. Create Document Chain
# ---------------------------
from langchain.chains.combine_documents import create_stuff_documents_chain

document_chain = create_stuff_documents_chain(
    llm=llm,
  ---------------------------
from langchain.chains import create_retrieval_chain

retrieval_chain = create_retrieval_chain(
    retriever=retriever,
    document_chain=document_chain
)   prompt=prompt
)


# ---------------------------
# 6. Retrieval Chain
#


# ---------------------------
# 7. Run a Query
# ---------------------------
response = retrieval_chain.invoke({
    "input": "Explain Transformers."
})

print("\n--- Answer ---\n")
print(response["answer"])
```

---

## 🧪 How to Use

1. Put your PDF in the project folder.
2. Update the filename inside the script:

```python
loader = PyPDFLoader("attention.pdf")
```

3. Run the program:

```bash
python app.py
```

4. Output will look like:

```
--- Answer ---

Transformers are neural network architectures based on self-attention...
```

---

## 📦 Project Structure

```
├── app.py
├── attention.pdf
├── README.md
├── requirements.txt
└── .env
```

---

## 🧭 How It Works (Architecture)

1. **PDF Loader** → Extracts text
2. **Text Splitter** → Breaks text into manageable overlapping chunks
3. **Embeddings** → Converts chunks into vectors
4. **FAISS Vector Store** → Enables fast semantic search
5. **Retriever** → Fetches matching chunks
6. **ChatOpenAI (GPT)** → Answers using only retrieved context
7. **RAG Chain** → Links retrieval + generation

---

## 🔮 Future Improvements

* Add a Streamlit UI
* Add chat history memory
* Add rerankers (Cohere, Jina)
* Add local LLMs via Ollama
* Build a LangGraph version

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

