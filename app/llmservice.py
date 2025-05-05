from typing import List, Optional
from app.model_factory.factory import get_model
from app.DBServices.chroma_writer import ChromaWriter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
import requests
from fastapi import HTTPException
from app.app_config import CONFIG

API_URL = "https://router.huggingface.co/fireworks-ai/inference/v1/chat/completions"
headers = {"Authorization": f"Bearer {CONFIG.HUGGING_FACE_API_KEY}"}

class LLMService:
    def __init__(self, provider="huggingface", model_name=None):
        self.model_name = model_name
        self.provider = provider
        self.vector_db = self._load_vector_db()
        self.trusted_sources = ["nytimes.com", "bbc.com", "investopedia.com"]  # Example trusted sources
        self.TAVILY_API_KEY = CONFIG.TAVILY_API_KEY
        self.TAVILY_API_URL = CONFIG.TAVILY_API_URL 

    def _load_vector_db(self):
        print("Loading Chroma DB... ✅")
        return ChromaWriter(
            persist_directory="./chroma_data",
            embedding_function=get_model(
                "embedding", "sentence-transformers/all-MiniLM-L6-v2", "huggingface"
            )
        )

    def generate_response(self, query: str) -> str:
        if not self.vector_db:
            raise ValueError("Vector database (Chroma) is not loaded.")

        print(f"Querying Chroma DB with: {query}")
        docs = self.vector_db.get_embeddings(query, k=3)
        context = "\n".join([doc.page_content[:500] for doc in docs])
        prompt = (
            "You are a helpful assistant. Use the following context (if available) to answer the user's question accurately. And do not use web search "
            "If the context does not contain the answer, rely on your own knowledge or web results. "
            "Make it clear when your answer is based on external information (e.g., say 'Based on web search...' or 'From my general knowledge...').\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
        return self._call_hf_chat([{"role": "user", "content": prompt}])

    def generate_response_remote(self, query: str) -> str:
        print("Falling back to plain LLM...")
        return self._call_hf_chat([{"role": "user", "content": query}])

    def _call_hf_chat(self, messages) -> str:
        payload = {
            "messages": messages,
            "model": "accounts/fireworks/models/deepseek-v3-0324",
            "max_tokens": 512
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise ValueError(f"Hugging Face API failed with status {response.status_code}")

    def grade_web_results(self, query: str, results: List[dict]) -> List[dict]:
        graded_results = []
        
        for result in results:
            # Basic similarity score (Cosine Similarity or other methods)
            similarity_score = self.compute_similarity(query, result['snippet'])
            
            # Source trustworthiness
            trustworthiness_score = 0
            if any(source in result['source'] for source in self.trusted_sources):
                trustworthiness_score = 1  # trusted source

            #  Content freshness
            freshness_score = 0
            if 'date' in result and self.is_recent(result['date']):
                freshness_score = 1
            
            # Combine all scores into an overall score
            overall_score = (similarity_score * 0.4) + (trustworthiness_score * 0.4) + (freshness_score * 0.2)
            
            graded_results.append({
                "snippet": result['snippet'],
                "source": result['source'],
                "score": overall_score
            })
        
        #Sort results by score and return top results
        graded_results.sort(key=lambda x: x['score'], reverse=True)
        return graded_results

    def compute_similarity(self, query: str, snippet: str) -> float:
        #we can implement a more sophisticated similarity measure here
        # For now,using a simple keyword match or cosine similarity
        return 0.8  # Placeholder

    def is_recent(self, date_str: str) -> bool:
        # need to Implement logic to check if the content is recent
        return True  # Placeholder

    def fetch_web_answer(self, query: str) -> Optional[str]:
        try:
            # Construct the request payload for Tavily's search API
            payload = {
                "query": query,
                "api_key": self.TAVILY_API_KEY  # Tavily API key
            }
            
            # Call the Tavily API
            response = requests.post(self.TAVILY_API_URL, json=payload)
            if response.status_code == 200:
                results = response.json().get('results', [])
                if results:
                    # Grade the results
                    graded_results = self.grade_web_results(query, results)
                    if graded_results:
                        best_result = graded_results[0]  # Get the highest-scoring result
                        return best_result['snippet']  
                return None
            else:
                print(f"Error fetching from Tavily: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching or grading web search results from Tavily: {e}")
            return None
