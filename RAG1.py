# 1. Extract PDF text

from pypdf import PdfReader

def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

raw_text = extract_pdf_text("PDF_Path")


# 2. Clean the text

import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    stop_words = set(stopwords.words("english"))
    cleaned = " ".join([word for word in text.split() if word not in stop_words])

    return cleaned

cleaned_text = preprocess_text(raw_text)


# 3. Chunk the text

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

chunks = chunk_text(cleaned_text)
print("Chunks created:", len(chunks))


# 4. Create FAISS Index

import faiss
import numpy as np

embeddings = model.encode(chunks, convert_to_numpy=True).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("FAISS DB Ready!")
print("Number of vectors stored:", index.ntotal)




# 5. Prompt Template

def build_prompt(context, query):
    prompt = f"""
You are an AI assistant that must answer ONLY using the context below.
If the answer is NOT in the context, you MUST reply:

"I don't know. The document does not contain this information."

---
CONTEXT:
{context}
---

USER QUESTION:
{query}

RULES:
- DO NOT use outside knowledge
- DO NOT guess
- Answer ONLY if the context contains the answer

ANSWER:
"""
    print(prompt)
    return prompt


def get_top_chunks(query, k=3):
    query_emb = model.encode([query], convert_to_numpy=True).astype("float32")

    distances, indices = index.search(query_emb, k)

    return [chunks[i] for i in indices[0]]



# 6. GROQ LLM Integration

import os
from groq import Groq

# Store your API key in environment variables for safety
# In CMD: setx GROQ_API_KEY "your_key_here"
client = Groq(api_key="YOUR_API_KEY")


def generate_answer(query):
    top_chunks = get_top_chunks(query)
    context = "\n\n".join(top_chunks)

    prompt = build_prompt(context, query)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.1
    )

    return response.choices[0].message.content
print(query)
print(embeddings)

# 7. Chat 

print("RAG Chatbot Ready using Groq LLM!\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    answer = generate_answer(user_input)
    print("\nBot:", answer, "\n")
