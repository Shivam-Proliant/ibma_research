import requests
import psutil
import os
import time

# --- Port Configuration ---
REASONING_URL = "http://localhost:8080/v1/chat/completions" # Gemma-3 27B
EMBEDDING_URL = "http://localhost:8081/v1/embeddings"      # EmbeddingGemma-300M

def get_llm_response(prompt, system_prompt="You are a helpful AI."):
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 150,
        "stream": False # Keep False for simplicity, but increase timeout
    }
    try:
        # Increased timeout to 600s for heavy 27B reasoning
        response = requests.post(REASONING_URL, json=payload, timeout=600)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.Timeout:
        return "Error: Reasoning model (8080) timed out after 10 minutes."
    except Exception as e:
        return f"Error: {e}"

def get_embedding(text):
    """Note: EmbeddingGemma-300M outputs 768 dimensions."""
    if not text or str(text).lower() == "nan" or str(text).strip() == "":
        return [0.0] * 768 
    
    payload = {"input": text}
    try:
        response = requests.post(EMBEDDING_URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        # Handle standard llama.cpp / OpenAI format
        if 'data' in data:
            return data['data'][0]['embedding']
        return data['embedding']
    except Exception as e:
        print(f"Error connecting to Embedding-300M (8081): {e}")
        return [0.0] * 768

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)
