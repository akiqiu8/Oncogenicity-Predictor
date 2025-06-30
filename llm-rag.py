import replicate
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv("secrets.env")

# 1. Create a simple knowledge base
documents = [
  {"text":"Academy award winner in year 2025 was 'Anora'.","metadata":{"citation":"ref2025"}},
  {"text":"Academy award winner in year 2024 was 'Oppenheimer'.","metadata":{"citation":"ref2024"}},
  {"text":"Academy award winner in year 2023 was 'Everything Everywhere All at Once'.","metadata":{"citation":"ref2023"}},
  {"text":"Academy award winner in year 2022 was 'Bobs Story: Becoming an Informaticist'.","metadata":{"citation":"ref2022"}},
  {"text":"Academy award winner in year 2021 was 'Nomadland'.","metadata":{"citation":"ref2021"}}
]

# 2. Embed the knowledge base
model_name = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(model_name)
# doc_embeddings = embedder.encode(documents, convert_to_numpy=True)
doc_embeddings = embedder.encode(documents, convert_to_numpy=True, normalize_embeddings=True)

# 3. Build FAISS index
dimension = doc_embeddings.shape[1]
#index = faiss.IndexFlatL2(dimension)
index = faiss.IndexFlatIP(dimension) # use cosine similarity
index.add(doc_embeddings)

# 4. Interactive chat loop
client = replicate.Client(api_token=os.getenv('REPLICATE_API_TOKEN'))
model = "meta/meta-llama-3-8b-instruct"
#model = "openai/gpt-4o"

chat_history = []

print("Welcome to the RAG chatbot! Type 'exit' or 'quit' to end the session.\n")
while True:
    query = input("You: ")
    if query.lower() in ("exit", "quit"):
        print("Goodbye!")
        break

    # Embed the user query
    # query_embedding = embedder.encode([query], convert_to_numpy=True)
    query_embedding = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    # Retrieve relevant documents
    threshold = 1
    k = 1
    D, I = index.search(query_embedding, k)
    print(f"Retrieved {k} documents with distances: {D[0]} and indices: {I[0]}")
    if D[0][0] <= threshold:
        retrieved_doc = documents[I[0][0]]
        retrieved_context = retrieved_doc["text"]
        retrieved_citation = retrieved_doc["metadata"]["citation"]
    else:
        retrieved_doc = {}
        retrieved_context = "" 
        retrieved_citation = ""

    print(f"Retrieved content: {retrieved_context}")

# Format chat history
    history_str = ""
    for turn in chat_history:
        history_str += f"User: {turn['question']}\nBot: {turn['answer']}\n"

    # Augment the prompt
    augmented_prompt = (
        f"{history_str}"
        f"Context: {retrieved_context}\n"
        f"Citation: {retrieved_citation}\n\n"
        f"Question: {query}\n"
        f"Please answer the question using the context if provided. \
            Include the citation in your response if it is available.\
            If citation is not available, do not mention that it is not available. \
            You do not need to prefix your response with 'According to the context' or similar phrases.\n\n"
    )

    # Call Replicate LLM
    output = client.run(
        model,
        input={
            "prompt": augmented_prompt,
            "temperature": 0.7,
            "max_new_tokens": 300
        }
    )

    response = "".join(output).lstrip()
    print(f"Bot: {response}\n")

    # Save this turn to history
    chat_history.append({"question": query, "answer": response})