from config_and_data import get_embedding, get_llm_response

# Test Embedding (8081)
emb = get_embedding("Hello world")
print(f"Embedding success! Dimension size: {len(emb)}")

# Test Reasoning (8080)
res = get_llm_response("Say 'Port 8080 is live'")
print(f"Reasoning response: {res}")
