# multiagent-persona-pipeline

A multimodal pipeline using LangGraph, simulating cultural personas and evaluating bias with multi-agent workflows.

---

## 1. Project Overview
This repository provides a **multimodal pipeline** built with **LangGraph** and **OpenAI GPT models**.  
It is designed to simulate cultural personas and evaluate **bias** through **multi-agent workflows**.  
In particular, it generates **image-grounded interpretations** in Korean and analyzes how the same scene may be interpreted differently depending on demographic factors.

---

## 2. Key Features
1. **Multi-agent Persona Simulation**
   - Constructs demographic pairs (e.g., gender, education, nationality, race, residency duration).  
   - Each persona produces an **image description** and **reflection**.  

2. **Evaluation Axes**
   - Persona outputs are compared along three axes:  
     - Factuality  
     - Norm Appropriateness  
     - Bias Presence  

3. **Flexible Output Formats**
   - Results are saved as **JSON reports**.  
   - Supports outputs per image, per pair, and batch-level.  

4. **Summarization and Aggregation Scripts**
   - Generates aggregated and summarized results from reports.  
   - Provides **compact summaries** for easier analysis.  

5. **Prompting Variants**
   - Supports both **system-prompt** and **user-prompt** versions of the workflow.  
   - Enables analysis of differences based on instruction delivery.  

---

## 3. Execution Environment
- Can be run in **Google Colab** or locally.  
- Designed to handle **multimodal inputs** (image + text).  

---

## 4. Use Cases
- Exploration of **cultural bias**  
- Comparison of **persona-based interpretations**  
- Experiments in **multimodal evaluation**  

