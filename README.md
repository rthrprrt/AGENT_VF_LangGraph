# AGENT_VF_LangGraph

Autonomous AI agent for professional thesis generation using LangGraph.

This project aims to develop an AI agent, AGENT_VF_LangGraph, capable of autonomously writing a complete professional thesis (Epitech "Mission Professionnelle" report). The agent leverages local LLMs (via Ollama), local vector stores, and the LangGraph framework for orchestrating complex, multi-step reasoning and generation processes.

## Project Vision

*   **Automated Thesis Generation:** Generate a comprehensive, well-structured, and coherent professional thesis.
*   **Input Sources:**
    *   School Deliverables/Guidelines (Epitech PDF).
    *   Apprenticeship Journal/Work Backlog (primary raw material).
*   **Output:** A complete thesis document adhering to school guidelines.
*   **Core Principles:** Modularity, Robustness, Local First, Traceability, Iterative Refinement (HITL).

## Tech Stack (Initial)

*   Python 3.11+
*   LangGraph
*   LangChain
*   Ollama (with models like gemma2:9b or similar)
*   FAISS (for local vector store)
*   Poetry (for dependency management)
*   Ruff (for linting and formatting)

## Setup

1.  Ensure Python 3.11+ and Poetry are installed.
2.  Clone the repository:
    ```bash
    git clone https://github.com/rthrprrt/AGENT_VF_LangGraph.git
    cd AGENT_VF_LangGraph
    ```
3.  Install dependencies:
    ```bash
    poetry install
    poetry install --group dev
    ```
4.  Activate the virtual environment:
    ```bash
    poetry shell
    ```
5.  Set up necessary environment variables (e.g., in a `.env` file):
    ```
    # Example .env content
    OLLAMA_BASE_URL="http://localhost:11434"
    # Add other configurations as needed
    ```

## Development

(Details to be added as development progresses)

## Roadmap Phases

*   **Phase 0:** Fondation & Environnement
*   **Phase 1:** Ingestion et Vectorisation
*   **Phase 2:** Planification & Boucle de Section Initiale
*   **Phase 3:** Amélioration Rédaction & Auto-Critique
*   **Phase 4:** Compilation & Bibliographie
*   **Phase 5:** Tests Complets, Raffinements & Documentation

## License

(To be determined)