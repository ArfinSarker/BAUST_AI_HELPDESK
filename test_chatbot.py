import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from retrieval.rag_search import search_information
from llm.chatbot import generate_answer

questions = [
    "Hi",
    "BAUST এর ভিসি কে?",
    "CSE ডিপার্টমেন্টের হেড কে?",
    "ভর্তি হতে কত টাকা লাগবে?",
    "Write a quick Python function for binary search.",
    "সূর্যগ্রহণ কেন হয়?"
]

print("=== STARTING CHATBOT EVALUATION ===")
for q in questions:
    print("\n--------------------------------------------------")
    print(f"USER: {q}")
    ctx = search_information(q)
    ans = generate_answer(ctx, q)
    print(f"AI RESPONSE:\n{ans}")
print("\n=== EVALUATION COMPLETE ===")
