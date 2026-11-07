# SOXGuard-AI
This script implements an AI-powered Governance, Risk, and Compliance (GRC) Auditor built using Streamlit, LangChain, and OpenAI. It automates the review of corporate financial documents against the Sarbanes-Oxley (SOX) Act baseline.
<div align="center">

# 🛡️ SOXGuard AI: Enterprise GRC Compliance Auditor

**An AI-powered Governance, Risk, and Compliance (GRC) automation agent built with Streamlit, LangChain, and OpenAI.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white)](https://openai.com/)

</div>

---

## 🚀 Overview

**SOXGuard AI** is an advanced Retrieval-Augmented Generation (RAG) and structured extraction pipeline designed to automate the review of corporate financial documents against the Sarbanes-Oxley (SOX) Act baseline. 

It evaluates Business Requirements Documents (BRDs) and SEC 10-K extracts, performs strict relevance validation, generates a four-quadrant SWOT compliance matrix, and supports stateful multi-turn follow-up conversations.

---

## 🏛️ Architecture & Core Features

* **🛡️ Relevance Gate Guardrail:** Powered by Pydantic schema enforcement (`ComplianceSWOT`), the agent detects invalid or unrelated uploads (e.g., resumes, personal files, or gibberish) and safely zeroes out compliance ratings instead of forcing erroneous data into audit categories.
* **🔍 Dynamic RAG Pipeline:** Leverages `RecursiveCharacterTextSplitter`, OpenAI embeddings (`text-embedding-3-large`), and an in-memory **Chroma** vector store to index and retrieve precise chunks tailored to custom audit focus areas.
* **📊 Comprehensive SWOT Matrix:** Automatically maps findings into Strengths, Weaknesses, Opportunities, and Threats relative to standard SOX controls (Section 404 compliance, Segregation of Duties, immutable audit trails, and RBAC).
* **💬 Stateful Conversational Memory:** Implements LangChain's `ChatMessageHistory` and `RunnableWithMessageHistory` to allow interactive, context-aware follow-up questioning directly within the Streamlit UI.

---

## 🛠️ Tech Stack

* **Frontend & UI:** Streamlit
* **Orchestration & LLM:** LangChain, OpenAI GPT-4o-mini
* **Validation & Parsing:** Pydantic
* **Vector Database:** ChromaDB
* **Document Loading:** PyPDFLoader

---

## ⚙️ Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/navjotrandhawa/SOXGuard-AI.git](https://github.com/navjotrandhawa/SOXGuard-AI.git)
   cd SOXGuard-AI

   
