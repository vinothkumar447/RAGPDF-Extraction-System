# 📘 RAG PDF Extraction System

A Retrieval-Augmented Generation (RAG) system that extracts text from
PDFs, chunks the content, generates embeddings, stores them in a vector
database, and answers user queries accurately using LLMs.

## 🚀 Project Features

-   Extract text from PDFs\
-   Chunk documents\
-   Generate embeddings\
-   Store embeddings in FAISS\
-   Query-answering with RAG\
-   Print query embeddings

## 📂 Project Structure

    rag-pdf-extraction/
    │── CSR MODULES.pdf
    │── RAG1.py
    │── requirements.txt
    │── README.md

## 🛠 Libraries Used

  Library              -->   Purpose
    
  PyPDF2               -->  Extract PDF text
  
  LangChain            -->  RAG pipeline
  
  SentenceTransformers -->  Embeddings
  
  FAISS               -->   Vector DB
  
  dotenv              -->   Load API keys

## ▶️ How to Run

``` bash
python RAG1.py
```

## 📌 Output Example

    Answer: The purpose of the document is...
