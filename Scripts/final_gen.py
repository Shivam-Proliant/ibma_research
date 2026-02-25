import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- IBMA Constants ---
DECAY_RATE = 0.01
SLASH_THRESHOLD = 0.5
CYCLES = 5000
SLASH_INTERVAL = 500

class IBMASimulator:
    def __init__(self):
        self.nodes = [] # List of [Vitality, Weight]
        self.stats = []

    def run_simulation(self):
        print("Starting Homeostasis Simulation...")
        for t in range(1, CYCLES + 1):
            # 1. Ingest new node [V=1.0, W=0.0]
            self.nodes.append([1.0, 0.0])
            
            # 2. Simulate random retrieval (Reinforce 5 random nodes)
            if len(self.nodes) > 5:
                indices = np.random.choice(len(self.nodes), 5, replace=False)
                for idx in indices:
                    self.nodes[idx][1] += 0.2 

            # 3. Apply Decay
            for n in self.nodes:
                n[0] *= np.exp(-DECAY_RATE)

            # 4. Slashing Phase
            if t % SLASH_INTERVAL == 0:
                nodes_before = len(self.nodes)
                self.nodes = [n for n in self.nodes if (n[0] + n[1]) >= SLASH_THRESHOLD]
                nodes_deleted = nodes_before - len(self.nodes)
                self.stats.append({
                    "turn": t, 
                    "active_nodes": len(self.nodes), 
                    "pruned": nodes_deleted
                })
        return pd.DataFrame(self.stats)

# --- Plotting Functions ---

def plot_stability(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df['turn'], df['active_nodes'], marker='o', color='#2ca02c', label='IBMA Active Nodes')
    plt.fill_between(df['turn'], df['active_nodes'], color='#2ca02c', alpha=0.1)
    plt.axhline(y=df['active_nodes'].mean(), color='r', linestyle='--', label='Average Homeostasis')
    plt.xlabel('Turns (Time)')
    plt.ylabel('Memory Footprint (Node Count)')
    plt.title('Figure 1: IBMA Homeostasis - Node Count Stabilization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('stability_plot.png')
    print("Saved stability_plot.png")

def plot_metabolic_state_space():
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    v = np.linspace(0, 1, 20)
    w = np.linspace(0, 1, 20)
    V, W = np.meshgrid(v, w)
    # Utility = Similarity + V + W (Abstracted)
    Utility = (V * 0.4) + (W * 0.6) 

    surf = ax.plot_surface(V, W, Utility, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_xlabel('Vitality (V)')
    ax.set_ylabel('Lignification (W)')
    ax.set_zlabel('Retrieval Utility')
    plt.title("Figure 2: Metabolic State-Space of Memory Instances")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.savefig('state_space_3d.png')
    print("Saved state_space_3d.png")

# --- Execution ---
if __name__ == "__main__":
    sim = IBMASimulator()
    df_stats = sim.run_simulation()
    
    plot_stability(df_stats)
    plot_metabolic_state_space()
    
    print("\n✅ All diagrams generated. You can now scp these .png files to your Windows machine for the paper.")
