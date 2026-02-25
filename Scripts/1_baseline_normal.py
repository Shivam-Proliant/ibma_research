import pandas as pd
import time
from config_and_data import get_llm_response, get_memory_mb

MAX_HISTORY = 10 
datasets_to_run = ["eval_musique.csv", "eval_narrativeqa.csv", "eval_babilong.csv"]

for dataset_file in datasets_to_run:
    print(f"\n🚀 Running Baseline on: {dataset_file}")
    try:
        df = pd.read_csv(dataset_file)
    except Exception: continue

    context_history = []
    results = []

    for index, row in df.iterrows():
        # Handle Potential NaN values from CSV
        fact = str(row['fact']) if not pd.isna(row['fact']) else ""
        query = str(row['query']) if not pd.isna(row['query']) else "None"
        expected = str(row['expected'.replace('.0', '')]) if not pd.isna(row['expected']) else "None"
        
        context_history.append(fact)
        if len(context_history) > MAX_HISTORY:
            context_history.pop(0)

        # STRICT CHECK: Only call LLM if there is a real question
        if query.lower() != "none" and query.strip() != "":
            prediction = get_llm_response(f"Context: {' | '.join(context_history)}\n\nQuestion: {query}")
            is_correct = expected.lower() in prediction.lower()
            status = "✅ CORRECT" if is_correct else "❌ WRONG"
        else:
            prediction = "SKIPPED"
            is_correct = None
            status = "📥 Storing"

        results.append({
            "turn": row['turn'], "is_correct": is_correct, "memory_mb": get_memory_mb()
        })
        
        if is_correct is not None:
            print(f"  Turn {row['turn']} | {status} | Expected: {expected} | Got: {prediction[:20]}...")
        elif row['turn'] % 100 == 0:
            print(f"  Turn {row['turn']} | {status} (Silent processing...)")

    pd.DataFrame(results).to_csv(f"results_baseline_{dataset_file}", index=False)
