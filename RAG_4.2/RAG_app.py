import logging
import warnings
from transformers import logging as hf_logging

logging.getLogger("langchain.text_splitter").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

#3.2
import os

import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

#3.3
chunk_size = 300
chunk_overlap = 30
model_name = "sentence-transformers/all-distilroberta-v1"
top_k = 20

cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
top_m = 8

#3.4
with open("Selected_Document.txt", "r", encoding="utf-8") as file:
    text = file.read()

#3.5
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)

chunks = text_splitter.split_text(text)

#3.6
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer(model_name)
embeddings = embedder.encode(chunks, show_progress_bar=False)

embedding_array = np.array(embeddings, dtype=np.float32)

faiss_index = faiss.IndexFlatL2(embedding_array.shape[1])
faiss_index.add(embedding_array)

#3.7
def retrieve_chunks(question, k=top_k):
    q_vec = embedder.encode([question], show_progress_bar=False)
    q_arr = np.array(q_vec, dtype=np.float32)

    _, I = faiss_index.search(q_arr, k)

    return [chunks[i] for i in I[0] if 0 <= i < len(chunks)]

#3.8
from sentence_transformers import CrossEncoder

reranker = CrossEncoder(cross_encoder_name)


def dedupe_preserve_order(items):
    seen = set()
    deduped = []

    for item in items:
        normalized = " ".join(item.split())
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)

    return deduped


def rerank_chunks(question: str, candidate_chunks: list[str], m: int = top_m) -> list[str]:
    pairs = [(question, chunk) for chunk in candidate_chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidate_chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    top_chunks = [chunk for chunk, _ in ranked[:m]]
    return dedupe_preserve_order(top_chunks)

#3.9
def build_user_prompt(context: str, question: str) -> str:
    return f"""Context:
{context}

Question: {question}

Answer:"""


def answer_question(question):
    """Retrieve, rerank, and answer a question using the selected context."""
    candidates = retrieve_chunks(question)
    relevant_chunks = rerank_chunks(question, candidates, m=top_m)
    context = "\n\n".join(relevant_chunks)

    system_prompt = (
        "You are a knowledgeable assistant that answers questions based on "
        "the provided context. If the answer is not in the context, say you don’t know."
    )

    user_prompt = build_user_prompt(context, question)

    resp = openai.chat.completions.create(
        model="gpt-5.4-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_completion_tokens=500
    )

    return resp.choices[0].message.content.strip()

#3.10
if __name__ == "__main__":
    print("Enter 'exit' or 'quit' to end.")
    while True:
        question = input("Your question: ")
        if question.lower() in ("exit", "quit"):
            break
        print("Answer:", answer_question(question))

#4.1


#4.2