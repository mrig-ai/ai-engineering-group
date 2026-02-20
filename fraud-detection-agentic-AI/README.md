# Fraud Detection Explainability Agent

This project implements an agentic fraud detection pipeline using **LangGraph** and **Large Language Models (LLMs)**. The system doesn't just predict if a transaction is fraudulent; it provides detailed, human-readable explanations and generates professional investigation reports in PDF format.

## 🚀 Key Features

- **Agentic Orchestration**: Uses `LangGraph` to manage a stateful workflow across multiple specialized nodes.
- **Explainable AI (XAI)**: Integrates SHAP values to identify and explain the specific features driving risk predictions.
- **Structured LLM Analysis**: Employs `Pydantic` models to ensure the LLM generates consistent, audit-ready investigation summaries.
- **PDF Report Generation**: Automatically transforms the final analysis into a professional PDF report.

## 🏗️ Architecture

The pipeline follows a directed graph flow:
1. **Data Processing**: Cleans and transforms raw transaction data.
2. **Model Inference**: Invokes a trained ML model for risk scoring.
3. **Model Explainability**: Calculates feature importance using SHAP.
4. **LLM Analyst**: Generates a human-friendly narrative of the findings.
5. **PDF Generation**: Saves the results as `transaction_report.pdf`.

## 📁 Project Structure

- `explainability_agent.ipynb`: The main notebook containing the agent logic and documentation.
- `scripts/utils.py`: Utility functions for data processing, prediction, and PDF generation.
- `artifacts/`: Serialized model and feature metadata.  
- `scripts/prompts.txt`: System instructions for the LLM Fraud Investigation Analyst.  
- `data/`: Directory containing sample transaction data for inference.  

## 🛠️ Setup

1. **Environment Variables**: Create a `.env` file with your `OPENAI_API_KEY`.
2. **Dependencies**: Install the required packages:
   ```bash
   uv sync
   ```
3. **Run**: Open `explainability_agent.ipynb` and execute the cells to run the agent on sample data.

