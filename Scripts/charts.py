import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication-style aesthetics
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

def plot_memory_growth():
    # Using LanceDB data to show Linear Bloat vs Baseline LLM
    lancedb_df = pd.read_csv("results_lancedb_synthetic_world_state.csv")
    baseline_df = pd.read_csv("results_baseline_synthetic_world_state.csv")
    
    plt.figure(figsize=(8, 5))
    plt.plot(lancedb_df['turn'], lancedb_df['memory_mb'], label='LanceDB (Static Vector RAG)', color='red', linewidth=2)
    plt.plot(baseline_df['turn'], baseline_df['memory_mb'], label='Normal LLM (Truncated Context)', color='gray', linestyle='--')
    
    plt.title('Memory Consumption: The Problem of Unpruned RAG')
    plt.xlabel('Conversational Turn')
    plt.ylabel('Memory Footprint (MB)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig2_memory_bloat.png', dpi=300)
    plt.show()

def plot_ibma_homeostasis():
    # Using the ibma_stability_data.csv
    data = {
        'turn':[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],
        'active_nodes':[415, 637, 802, 935, 1042, 1135, 1222, 1276, 1346, 1400]
    }
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(8, 5))
    plt.plot(df['turn'], df['active_nodes'], marker='o', color='green', linewidth=2, label="IBMA Active Nodes")
    plt.axhline(y=1500, color='r', linestyle='--', label="Asymptotic Memory Ceiling")
    
    plt.fill_between(df['turn'], df['active_nodes'], color='green', alpha=0.1)
    plt.title('IBMA Homeostasis: Node Count Stabilization via Slashing')
    plt.xlabel('Conversational Turn (Time)')
    plt.ylabel('Active Nodes in Knowledge Graph')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig3_ibma_homeostasis.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_memory_growth()
    plot_ibma_homeostasis()
