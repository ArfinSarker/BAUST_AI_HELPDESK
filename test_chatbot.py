import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from retrieval.rag_search import search_information
from llm.chatbot import generate_answer, rewrite_query_with_history

history = []

conversations = [
    "tomar naam ki",
    "kemon acho",
    "koyta dept ache",
    "BAUST এর ভিসি কে?",
    "eita english e bolo",
    "Who is the Head of CSE department?"
]

print("=== STARTING MULTI-TURN LANGUAGE & TRANSLATION EVALUATION ===", flush=True)

for q in conversations:
    print("\n" + "="*60, flush=True)
    print(f"USER: {q}", flush=True)
    
    effective_q = rewrite_query_with_history(q, history)
    ctx = search_information(effective_q)
    ans = generate_answer(ctx, q, history=history)
    
    print(f"AI RESPONSE:\n{ans}", flush=True)
    history.append({"role": "user", "content": q})
    history.append({"role": "assistant", "content": ans})

print("\n=== EVALUATION COMPLETE ===", flush=True)

