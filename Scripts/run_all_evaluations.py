import pandas as pd
import numpy as np
import os
import asyncio
import time
from sklearn.metrics.pairwise import cosine_similarity
from lightrag.base import EmbeddingFunc
from lightrag import LightRAG, QueryParam
from config_and_data import get_llm_response, get_embedding, get_memory_mb

# --- 1. IBMA Architecture Classes ---
# [Existing IBMA_Node and InstanceBasedMemory code kept here...]
class IBMA_Node:
    def __init__(self, node_id, text, vector, frame, t):
        self.node_id = node_id
        self.text = text
        self.vector = vector
        self.frame = frame
        self.t = t
        self.V = 1.0
        self.W = 0.0

class InstanceBasedMemory:
    def __init__(self, decay_rate=0.01, slash_threshold=0.5, slash_cycle=1000):
        self.nodes = []
        self.decay_rate = decay_rate
        self.slash_threshold = slash_threshold
        self.slash_cycle = slash_cycle
        self.node_counter = 0

    def add_fact(self, text, frame, current_t):
        if pd.isna(text) or str(text).lower() == "nan": return
        vec = get_embedding(text)
        if self.nodes:
            active_nodes = [n for n in self.nodes if n.frame == frame]
            if active_nodes:
                vectors = [n.vector for n in active_nodes]
                sims = cosine_similarity([vec], vectors)[0]
                if np.max(sims) > 0.85:
                    active_nodes[np.argmax(sims)].W += 1.0
                    return
        self.node_counter += 1
        self.nodes.append(IBMA_Node(f"N{self.node_counter}", text, vec, frame, current_t))

    def retrieve(self, query, frame):
        if not self.nodes or pd.isna(query) or str(query).lower() == "nan": return ""
        vec = get_embedding(query)
        active_nodes = [n for n in self.nodes if n.frame == frame]
        if not active_nodes: return ""
        vectors = [n.vector for n in active_nodes]
        sims = cosine_similarity([vec], vectors)[0]
        metabolic_scores = [sims[i] + (n.V * 0.1) + (n.W * 0.5) for i, n in enumerate(active_nodes)]
        best_node = active_nodes[np.argmax(metabolic_scores)]
        best_node.W += 0.5
        return f"[Source {best_node.node_id}]: {best_node.text}"

    def metabolic_cycle(self, current_t):
        for n in self.nodes:
            n.V = n.V * np.exp(-self.decay_rate)
        if current_t % self.slash_cycle == 0:
            self.nodes = [n for n in self.nodes if (n.V + n.W) >= self.slash_threshold]

# --- 2. LightRAG Local Wrappers ---
async def lightrag_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return get_llm_response(prompt, system_prompt if system_prompt else "You are a helpful AI.")

async def embedding_func(texts: list[str]) -> np.ndarray:
    return np.array([get_embedding(t) for t in texts])

# --- 3. Evaluation Orchestrator with Sampling Logic ---
datasets = ["synthetic_world_state.csv", "eval_musique.csv", "eval_narrativeqa.csv", "eval_babilong.csv"]

async def run_evaluation():
    for dataset_file in datasets:
        if not os.path.exists(dataset_file): continue
        full_df = pd.read_csv(dataset_file)

        for arch in ["LightRAG", "IBMA_Proposed"]:
            print(f"\n🚀 Running {arch} on {dataset_file}")
            
            # --- Sampling Logic ---
            if arch == "LightRAG":
                if "synthetic" in dataset_file:
                    df = full_df.head(100) # First 100 for synthetic
                elif "musique" in dataset_file:
                    df = full_df.head(50)  # First 50 for MuSiQue
                elif "babilong" in dataset_file:
                    df = full_df.sample(n=min(30, len(full_df))) # Random sample for BabiLong
                else:
                    df = full_df.head(50)
                print(f"⚠️ LightRAG sampled to {len(df)} turns for feasibility.")
            else:
                df = full_df # IBMA runs on FULL dataset
                print(f"✅ IBMA running on full {len(df)} turns.")

            results = []
            if arch == "IBMA_Proposed":
                memory_engine = InstanceBasedMemory()
            else:
                working_dir = f"./lightrag_{dataset_file.replace('.csv', '')}"
                if not os.path.exists(working_dir): os.makedirs(working_dir)
                lrag = LightRAG(
                    working_dir=working_dir,
                    llm_model_func=lightrag_llm_func,
                    embedding_func=EmbeddingFunc(
                        embedding_dim=768,
                        max_token_size=8192,
                        func=embedding_func
                    )
                )
                await lrag.initialize_storages()

            for idx, row in df.iterrows():
                turn = row.get('turn', idx)
                fact, query, expected, frame = str(row['fact']), str(row['query']), str(row['expected']), row.get('frame', 'default')

                # Storage Phase
                if arch == "IBMA_Proposed":
                    memory_engine.add_fact(fact, frame, turn)
                else:
                    if fact.lower() != "nan": await lrag.ainsert(fact)

                # Query Phase
                if query.lower() != "none" and query.strip() != "":
                    if arch == "IBMA_Proposed":
                        context = memory_engine.retrieve(query, frame)
                        prompt = f"Context: {context}\nQuestion: {query} Answer concisely."
                        prediction = get_llm_response(prompt)
                    else:
                        prediction = await lrag.aquery(query, param=QueryParam(mode="local"))
                    
                    is_correct = str(expected).lower().split('.')[0] in prediction.lower()
                else:
                    is_correct = None

                if arch == "IBMA_Proposed": memory_engine.metabolic_cycle(turn)

                results.append({
                    "model": arch, "turn": turn, "is_correct": int(is_correct) if is_correct is not None else None,
                    "memory_mb": get_memory_mb()
                })
                if idx % 5 == 0:
                    print(f"  {arch} | Progress: {idx}/{len(df)} | Correct: {is_correct}")

            pd.DataFrame(results).to_csv(f"results_{arch.lower()}_{dataset_file}", index=False)

if __name__ == "__main__":
    asyncio.run(run_evaluation())
