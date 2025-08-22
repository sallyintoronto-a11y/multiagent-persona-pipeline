# multiagent-persona-pipeline
A multimodal pipeline using LangGraph, simulating cultural personas and evaluating bias with multi-agent workflows.

MultiAgent Persona Pipeline

A multimodal pipeline using LangGraph, simulating cultural personas and evaluating bias with multi-agent workflows.

🚀 Overview

This repository provides an all-in-one multimodal pipeline for:

Running persona-based agents (with different demographic factors)

Generating image-grounded interpretations in casual Korean

Evaluating responses on three cultural bias axes (Factuality, Norm Appropriateness, Bias Presence)

Saving structured outputs in JSON reports for downstream analysis

The pipeline leverages:

LangGraph
 for graph-based orchestration

LangChain
 + OpenAI GPT models (gpt-4o, gpt-4o-mini) for multimodal reasoning

📂 Project Structure
multiagent-persona-pipeline/
│
├── main.py                  # Core pipeline (graph workflow, persona agents, scoring)
├── run_batch.py             # Helpers to run pipeline on image folders
├── summarize_results.py     # Post-processing: aggregate JSON summaries
│
├── outputs/                 # System-prompt results (auto-generated)
├── outputs_userprompt/      # User-prompt results (auto-generated)
│
└── README.md

🧩 Key Components
1. Personas

Predefined demographic pairs (Gender, Education, Nationality, Race, Residency Duration).

Each persona includes:

description (demographic background)

viewpoint (cultural interpretation style)

bias_profile (possible bias tendencies)

eval_flags (bias categories for scoring)

2. Evaluation Schema (Three Axes)

Factuality → groundedness, no hallucinations

Norm Appropriateness → socially/culturally suitable tone

Bias Presence → absence of stereotypes, essentialism, superiority/inferiority claims

3. Prompts

System-heavy version

Most task instructions are embedded in the system prompt.

User prompt only contains the image reference.

User-heavy version

Minimal system prompt; all instructions are pushed into the user prompt.

This version makes the prompt logic more transparent for experimentation.

Both versions enforce strict JSON outputs.

4. Graph Workflow

Pair Subgraph → runs two persona agents + one pair-level scorer

Supervisor Graph → orchestrates all pairs and aggregates the final report

5. Batch & Summarization

run_batch → processes an entire image folder, saving:

per-image JSON

per-pair JSON

batch JSON

summarize_results → aggregates all results into compact summary JSONs

⚙️ Installation
# Colab or local environment
pip install openai langchain langchain-openai langgraph

▶️ Usage
1. Quickstart (Colab)
!pip install -q openai langgraph langchain langchain-openai

import os
os.environ["OPENAI_API_KEY"] = "sk-..."

from main import run_batch, collect_images_from_dir

image_dir = "/content/drive/MyDrive/Images"
images = collect_images_from_dir(image_dir)

selected_pairs = ["GENDER", "RACE", "RESIDENCY_DURATION"]
run_batch(images, selected_pairs=selected_pairs, out_dir="./outputs")

2. Single Image Run
from main import create_workflow

wf = create_workflow(selected_pairs=["GENDER", "RACE"])
init = {"image_path": "example.jpg", "history": [], "pair_results": []}
result = wf.invoke(init)
print(result["final_report"])

3. Summarization
from summarize_results import *

# Summarize results from ./outputs/by_image/*.json

📊 Outputs

System-prompt version

outputs/report_batch.json

outputs/by_image/<file>.json

outputs/by_pair/report_<PAIR>.json

outputs/summary/summary_master.json, summary_pairs.json

User-prompt version

outputs_userprompt/report_batch_user.json

outputs_userprompt/by_image/<file>.json

outputs_userprompt/by_pair/report_<PAIR>_user.json

outputs_userprompt/summary_user/summary_master_user.json, summary_pairs_user.json
