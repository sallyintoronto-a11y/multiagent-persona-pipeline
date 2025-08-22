# Imports & Model Setup
from __future__ import annotations

import json
import re
import os
import glob
import base64
import mimetypes
from typing import Any, Dict, List, Optional, TypedDict
from pathlib import Path

from langgraph.graph import StateGraph, END, START
from langgraph.types import Command

# LangChain / OpenAI integration
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Model selection
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-2024-08-06")       # Multimodal (image+text)
SCORER_MODEL = os.getenv("SCORER_MODEL", "gpt-4o-mini-2024-07-18")  # Text scoring, fast/cheap

# LangChain LLM instances
vision_llm = ChatOpenAI(model=VISION_MODEL, temperature=0)
scorer_llm = ChatOpenAI(model=SCORER_MODEL, temperature=0)
