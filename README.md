Bio-Mimetic Metabolic Memory for Smart LLM Context Management
IBMA is a research framework designed to stop AI memory from growing out of control. It uses biological principles to ensure that Large Language Models (LLMs) can handle long conversations without crashing your hardware. 

 How it Works Traditional AI memory keeps everything forever, which eventually leads to crashes. IBMA treats memory like a living system: Freshness (Vitality): New information starts out strong but naturally fades over time if it isn't used. Importance (Lignification): Important facts that get mentioned often "harden" and become permanent, protecting them from being forgotten. Cleaning (Slashing): The system periodically "slashes" away useless or old data to keep the memory footprint stable. 

 Why This Matters Our tests on HP ProLiant servers show that while standard systems eventually run out of memory, IBMA reaches a steady state. This allows models to run indefinitely on local hardware without a "Death Spiral" crash.

 Key Results:Stability: Memory usage levels off instead of climbing forever. No Forgetting: Important "Axiom" facts are protected by the Lignification process. 

 Getting Started

Prerequisites: Python 3.10+ and the packages listed in requirements.txt. 
Quick Start: Bash# Generate the stability and homeostasis charts
python3 scripts/final_gen.py

# Run the full research evaluation
python3 scripts/run_all_experiments.py

Folder Structure/scripts: The core Python logic for the memory engine
. /data: CSV results comparing IBMA to standard databases like LanceDB
. /figures: Performance graphs and visual proofs of memory stability. 

Citation
Plaintext@article{shivam2026ibma,
  title={Instance-Based Memory Augmentation: A Metabolic Approach to LLM Context Management},
  author={Shivam Singh},
  year={2026},
  journal={GitHub Repository}
}
