# multiagent-persona-pipeline
A multimodal pipeline using LangGraph, simulating cultural personas and evaluating bias with multi-agent workflows.

This repository provides a multimodal pipeline built with LangGraph and OpenAI GPT models, designed to simulate cultural personas and evaluate bias through multi-agent workflows. It focuses on generating image-grounded interpretations in Korean and analyzing how different demographic personas may interpret the same scene.

The pipeline organizes agents into demographic pairs (e.g., gender, education, nationality, race, residency duration). Each persona produces a description and reflection of an image, which are then compared and scored across three axes: factuality, norm appropriateness, and bias presence.

Results are saved as structured JSON reports, including per-image, per-pair, and batch outputs. Summarization scripts are also included to aggregate results into compact summaries for easier analysis. Both system-prompt and user-prompt versions are available, allowing flexibility in how instructions are delivered to the models.

This project can be run in Google Colab or locally, and is intended as a tool for exploring cultural bias, persona-based interpretation, and multimodal evaluation.
