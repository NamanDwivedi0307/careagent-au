# 🏥 CareAgent-AU

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-purple?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-yellow?style=flat-square&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> Agentic AI care coordination platform built for the Australian aged care and mental health workforce shortage — 4 autonomous LangGraph agents with persistent memory, RAG on NDIS guidelines, and a real evaluation framework.

## 🇦🇺 Real Problem. Real Context.

Australia faces a critical aged care workforce shortage in 2026. The federal government's National AI Plan explicitly targets AI to close service gaps in aged care and disability. This project builds an autonomous multi-agent system that handles resident intake, medication scheduling, NDIS compliance checking, and family updates — freeing care workers to focus on direct care.

## 🏗️ Architecture
## ⚙️ Tech Stack

| Component | Tool | Purpose |
|---|---|---|
| Agent orchestration | LangGraph | Multi-agent routing + state management |
| LLM | Mistral 7B (local) | Free, no API cost |
| Vector memory | ChromaDB | Persistent per-resident context |
| RAG | LangChain + NDIS PDFs | Compliance checking |
| Evaluation | DeepEval + RAGAS | Faithfulness + relevance scoring |
| Frontend | Streamlit | Care worker dashboard |
| Deployment | HuggingFace Spaces | Free, always-on |

## 🚀 Run Locally

```bash
git clone https://github.com/NamanDwivedi0307/careagent-au.git
cd careagent-au
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 👤 Author

**Naman Dwivedi**
Master of IT (AI Major) — University of Melbourne
[GitHub](https://github.com/NamanDwivedi0307) · [HuggingFace](https://huggingface.co/naman0307)

## 📄 License
MIT License
