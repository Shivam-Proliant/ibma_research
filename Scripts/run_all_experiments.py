import pandas as pd
import numpy as np
import os
import asyncio
import time
from sklearn.metrics.pairwise import cosine_similarity
from config_and_data import get_llm_response, get_embedding, get_memory_mb

# --- 1. Dual-Process IBMA Engine ---
class IBMA_Node:
    def __init__(self, node_id, text, vector, frame, t):
        self.node_id = node_id
        self.text = text
        self.vector = vector
        self.frame = frame
        self.t = t
        self.V = 1.0  # Vitality (Active Memory)
        self.W = 0.0  # Lignification (Latent Memory)

class InstanceBasedMemory:
    def __init__(self, decay_rate=0.005, slash_threshold=0.3):
        self.nodes = []
        self.decay_rate = decay_rate
        self.slash_threshold = slash_threshold

    def add_fact(self, text, frame, current_t):
        if pd.isna(text) or str(text).strip() == "" or str(text).lower() == "nan": return
        vec = get_embedding(text)
        
        # Check for near-duplicates to reinforce existing nodes (Lignification)
        if self.nodes:
            active_nodes = [n for n in self.nodes if n.frame == frame]
            if active_nodes:
                vectors = [n.vector for n in active_nodes]
                sims = cosine_similarity([vec], vectors)[0]
                if np.max(sims) > 0.92:
                    idx = np.argmax(sims)
                    active_nodes[idx].W += 1.2 # Strong reinforcement to Latent memory
                    active_nodes[idx].V = 1.0   # Refresh Vitality
                    return

        self.nodes.append(IBMA_Node(f"N{len(self.nodes)}", text, vec, frame, current_t))

    def retrieve(self, query, frame):
        if not self.nodes: return ""
        q_vec = get_embedding(query)
        
        # Step 1: Filter by Frame with Global Fallback
        f_nodes = [n for n in self.nodes if n.frame == frame]
        if not f_nodes: f_nodes = self.nodes 
        
        vectors = [n.vector for n in f_nodes]
        sims = cosine_similarity([q_vec], vectors)[0]

        # Step 2: Scoring with Dual-Process Bias
        scored_nodes = []
        for i, n in enumerate(f_nodes):
            active_score = n.V * 0.2
            latent_score = min(n.W, 3.0) * 0.4 # Cap latent to avoid overshadowing
            total_score = sims[i] + active_score + latent_score
            scored_nodes.append((total_score, n))

        # Sort and take top 3 context pieces
        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        top_context = " ".join([node.text for score, node in scored_nodes[:3]])
        return top_context

    def metabolic_cycle(self):
        # Apply exponential decay to Vitality
        for n in self.nodes:
            n.V *= np.exp(-self.decay_rate)
        
        # Slashing: Only delete if the sum of Active and Latent strength is below threshold
        self.nodes = [n for n in self.nodes if (n.V + n.W) >= self.slash_threshold]

# --- 2. Robust Evaluation Runner ---
async def run_benchmark():
    datasets = ["eval_musique.csv", "eval_narrativeqa.csv", "eval_babilong.csv"]
    
    for d_file in datasets:
        if not os.path.exists(d_file): continue
        print(f"\n🧠 Testing IBMA (Dual-Process) on {d_file}...")
        
        df = pd.read_csv(d_file)
        memory = InstanceBasedMemory()
        results = []

        for idx, row in df.iterrows():
            fact = str(row.get('fact', ''))
            query = str(row.get('query', ''))
            expected = str(row.get('expected', ''))
            frame = str(row.get('frame', 'default'))

            # --- Storage ---
            memory.add_fact(fact, frame, idx)

            # --- Evaluation & Inference ---
            is_correct = False
            prediction = ""

            # Only evaluate if there is a real answer to find
            if query and query.lower() != "nan" and expected.lower() != "nan":
                context = memory.retrieve(query, frame)
                prompt = f"Context: {context}\nQuestion: {query}\nAnswer very concisely."
                prediction = get_llm_response(prompt)
                
                # Fuzzy match logic: check if expected keywords appear in LLM response
                is_correct = any(word.lower() in prediction.lower() for word in expected.split())
                
                # Record result for actual QA pairs
                results.append({
                    "turn": idx, 
                    "correct": int(is_correct), 
                    "nodes": len(memory.nodes),
                    "memory_mb": get_memory_mb()
                })
            
            # --- Decay ---
            memory.metabolic_cycle()

            if idx % 20 == 0:
                print(f"Turn {idx} | Nodes: {len(memory.nodes)} | Correct: {is_correct if expected.lower() != 'nan' else 'N/A (Noise)'}")

        pd.DataFrame(results).to_csv(f"results_IBMA_Dual_{d_file}", index=False)
        print(f"✅ Final results saved for {d_file}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
