import requests
import pandas as pd

def fetch_musique_rows(limit=100):
    print(f"🚀 Fetching first {limit} rows of MuSiQue via API...")
    
    # Hugging Face Dataset Viewer API Endpoint
    # This pulls raw data even if the .py loading script is broken.
    url = f"https://datasets-server.huggingface.co/rows?dataset=dgslibisey/MuSiQue&config=default&split=train&offset=0&length={limit}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        rows_data = response.json()['rows']
        
        parsed_data = []
        turn_counter = 1
        
        for item in rows_data:
            row = item['row']
            question = row['question']
            answer = row['answer']
            paragraphs = row['paragraphs']
            
            # 1. Inject the supporting facts first
            # We take only paragraphs marked as 'is_supporting' if available, 
            # otherwise just the first few.
            for p in paragraphs:
                if p.get('is_supporting', False) or paragraphs.index(p) < 3:
                    parsed_data.append({
                        "turn": turn_counter,
                        "fact": f"Context: {p['paragraph_text']}",
                        "query": "None",
                        "expected": "None",
                        "frame": "MuSiQue"
                    })
                    turn_counter += 1
            
            # 2. Add the reasoning question at the end of this block
            parsed_data.append({
                "turn": turn_counter,
                "fact": "--- Reasoning Task ---",
                "query": question,
                "expected": answer,
                "frame": "MuSiQue"
            })
            turn_counter += 1

        # Save to CSV
        df = pd.DataFrame(parsed_data)
        df.to_csv("eval_musique.csv", index=False)
        print(f"✅ Success! Created eval_musique.csv with {len(df)} turns.")

    except Exception as e:
        print(f"❌ Failed to fetch MuSiQue: {e}")

if __name__ == "__main__":
    fetch_musique_rows(50) # Pulls 50 questions (results in ~200 turns)
