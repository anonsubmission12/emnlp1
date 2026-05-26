# MERIT System

A sophisticated AI simulation where an Agent persuades a User to buy motor insurance, featuring a **10-year memory horizon**, **personality evolution**, and **continuous learning** via an LLM-as-a-Judge.

##  Key Features

*   **Closed-Loop Learning**: The system learns from every interaction. A Judge evaluates turns and updates the agent's strategy policy (Dense Graph).
*   **Long-Horizon Memory**:
    *   **5-Level Architecture**: Session, Strategy, Latent State, Persona Drift, and Concession Memory.
    *   **Persona Evolution**: Users "drift" over time (e.g., becoming more trusting or price-sensitive) based on past interactions.
*   **Spectrum-Aware Decision Making**:
    *   Decisions aren't just random; they are based on cosine similarity between the current context and potential strategies.
*   **Realistic Simulation**:
    *   **VLM Integration**: Users "see" and analyze car images.
    *   **Negotiation**: Agents and Users haggle over premiums with realistic constraints.
    *   **8 Distinct Personas**: From "Tech-Savvy" to "Cautious", each with dynamic Big Five traits.

---

## 📚 Documentation

Detailed documentation for the development team is available in the `docs/` (or root) folder:

1.  **[Master Guide (Start Here)](system_architecture_and_workflow.md)**: High-level architecture, data flow, and workflow.
2.  **[Memory Architecture](memory_architecture_details.md)**: Deep dive into the 5 memory levels and the `MemoryManager`.
3.  **[Graph Learning](graph_documentation.md)**: How the DenseGraph works and how weights are updated.
4.  **[Judge Metrics](judge_metrics.md)**: Explanation of the evaluation criteria (Coherence, Strategy Effectiveness, etc.).

---

## 🛠️ Setup

1.  **Install Dependencies**:
    ```bash
    pip install openai numpy scikit-learn tqdm python-dotenv
    ```

2.  **Configure API Key**:
    *   Create a `.env` file or set `OPENAI_API_KEY` in your environment.
    *   (Note: `openai_client.py` also supports a hardcoded key for local testing).

3.  **Verify Data**:
    *   Ensure `motor-insurance-updated.json` (Knowledge Base) is present.
    *   Ensure `memory/personas/` directory exists (created automatically).

---

## 🏃 Usage

### 1. Run the Simulation
To generate conversations (Default: 10-year simulation for a specific persona):

```bash
python main.py
```

*   **Test Mode**: Edit `main.py` and set `TEST_MODE = True` to generate a small batch (10 convos) for debugging.
*   **Full Run**: Set `TEST_MODE = False` for the full 120-conversation dataset generation.

### 2. View Outputs
Artifacts are saved to the `output/` directory:
*   `conversations.json`: The training dataset (Turn-by-turn logs).
*   `memory_snapshots.json`: How the agent's memory evolved.
*   `graph_weights.json`: The final "learned" policy.
*   `graphs/*.png`: Visualizations of the graph weights.

---

