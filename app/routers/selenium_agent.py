from fastapi import APIRouter, HTTPException
from app.services.chroma_service import ChromaService
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from app.models import EmbeddingQuery

# Initialize ChromaService
chroma_service = ChromaService()

router = APIRouter()

def fetch_embeddings_and_get_response(query_text: str, k: int = 5) -> str:
    """Fetch embeddings from ChromaDB and use Selenium to interact with dummy.com."""
    # Step 1: Get similar documents from ChromaDB
    documents = chroma_service.get_embeddings(query_text, k=k)
    
    # Step 2: Construct the context for dummy.com from the retrieved documents
    context = "\n".join([doc.page_content for doc in documents])
    
    # Step 3: Use Selenium to interact with dummy.com
    driver = webdriver.Chrome()  # Ensure appropriate ChromeDriver is available
    try:
        driver.get("http://dummy.com")  # URL for dummy.com
        
        # Wait for page to load (Adjust this as necessary)
        time.sleep(3)
        
        # Find the input field (Adjust the selector as per your actual page)
        input_field = driver.find_element(By.ID, "input_field")  # Replace with actual ID
        
        # Enter the context data into the input field
        input_field.send_keys(context)
        
        # Find the submit button and click it (Adjust the selector as needed)
        submit_button = driver.find_element(By.ID, "submit_button")  # Replace with actual ID
        submit_button.click()
        
        # Wait for the response (Adjust the sleep time or use WebDriverWait as per your need)
        time.sleep(5)
        
        # Capture the response (Adjust based on where the response appears)
        response_element = driver.find_element(By.ID, "response_field")  # Replace with actual ID
        response = response_element.text
        
        return response
    finally:
        driver.quit()  # Close the browser after the interaction


@router.post("/get-embeddings-response/")
async def get_embeddings_response(query: EmbeddingQuery):
    """Fetch embeddings from ChromaDB and query dummy.com for a response."""
    try:
        response = fetch_embeddings_and_get_response(query.query_text, k=query.k)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
