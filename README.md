# GenAI Quality Engineering Portfolio
**By Akash Raj** | Project Enginner(GenAI Engineer)

Welcome to my Quality Engineering Generative AI portfolio. This repository contains two Proof-of-Concept (POC) applications
built to bridge traditional software testing with emerging AI technologies. These tools are designed to solve real-world QA
bottlenecks: manual test creation time and test failure triage time.

## 🛠️ Tech Stack
* **Framework:** LangChain (LCEL)
* **Frontend:** Streamlit
* **Vector Database:** ChromaDB
* **Embeddings:** Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`)
* **LLMs:** * Google Gemini (`gemini-2.5-flash`)
* Hugging Face Qwen (`Qwen/Qwen2.5-7B-Instruct`)

---

## 🚀 Project 1: RAG Tosca Log Analyzer (`app.py`)

### Overview
A Retrieval-Augmented Generation (RAG) pipeline that reads raw, technical test execution logs (like Tricentis Tosca outputs)
and translates the stack traces into plain English root-cause analyses. 

### Features
* **Local Vector Search:** Uses ChromaDB to store and retrieve specific execution logs based on semantic similarity.
* **Contextual Accuracy:** Bypasses LLM hallucinations by forcing the AI to strictly base its answers on the retrieved `.txt` log files.
* **Source Transparency:** Includes a dropdown UI to display the exact source document the AI used to formulate its answer.

##🚀 Project 2: AI Auto-Test Case Generator (test_generator.py)
Overview

A dynamic prompt-engineering tool that converts Jira User Stories into comprehensive, structured manual test suites.
It automatically outputs Test Case IDs, Scenarios, Test Steps, and Expected Results for Positive, Negative, and Edge Cases.

###Features
* **Dynamic Domain Rules: Features a dropdown to inject specific business contexts into the AI. For example, selecting Lead to Order (L2O)
    forces the AI to structure test steps around a strict chronological orchestration plan (Account > Contact > Opportunity > Quote >
    Contract > Order > Workbench Submission > Orchestration Plan Activation).

* **Secure API Key Handling (BYOK): Utilizes a secure Streamlit sidebar input for the Hugging Face API key. The key is injected directly
   into the session environment memory, ensuring no secret keys are ever hardcoded or pushed to version control.
