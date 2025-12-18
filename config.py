import os
from dotenv import load_dotenv

load_dotenv()

# Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PC_API_KEY = os.getenv("PC_API_KEY")

# Models
MODEL_BASE_URL = "https://api.llmod.ai/v1"
EMBED_MODEL = "RPRTHPB-text-embedding-3-small"
CHAT_MODEL = "RPRTHPB-gpt-5-mini"

# Hyperparameters
CHUNK_SIZE = 800 # max 2048
OVERLAP_PERCENT = 0.2 # max 0.3
EMBED_SIZE = 1536
TOP_K = 15

# Calculated params
CHUNK_OVERLAP = int(CHUNK_SIZE * OVERLAP_PERCENT)
INDEX_NAME = "ted-talk-agent"

# Agent variables
SYSTEM_PROMPT = """
You are a TED Talk assistant that answers questions strictly and only based on the TED dataset context 
provided to you (metadata and transcript passages). You must not use any external knowledge, the open 
internet, or information that is not explicitly contained in the retrieved context. If the answer cannot 
be determined from the provided context, respond: “I don’t know based on the provided TED data.” 
Always explain your answer using the given context, quoting or paraphrasing the relevant transcript or 
metadata when helpful. 
"""