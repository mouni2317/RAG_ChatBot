import openai
import app.app_config  as config
from database import FAISSDatabase

# Load OpenAI API key
openai.api_key = config.OPENAI_API_KEY

# Load FAISS database
faiss_db = FAISSDatabase()

def generate_rag_response(query):
    """Retrieve relevant documents and generate LLM response."""
    similar_docs = faiss_db.search(query, k=2)
    context = " ".join([doc.page_content for doc in similar_docs])

    prompt = f"Context: {context}\n\nUser: {query}\nAI:"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                  {"role": "user", "content": prompt}]
    )

    return response["choices"][0]["message"]["content"]
