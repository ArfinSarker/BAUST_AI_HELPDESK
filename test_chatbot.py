import os
import sys
import time

# Auto-redirect to project virtual environment if run with global python
_venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "Scripts", "python.exe")
if os.path.exists(_venv_python) and os.path.abspath(sys.executable).lower() != os.path.abspath(_venv_python).lower():
    import subprocess
    sys.exit(subprocess.call([_venv_python] + sys.argv))

# Enable UTF-8 encoding for unicode and Bengali output
sys.stdout.reconfigure(encoding='utf-8')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable ANSI escape sequences on Windows consoles
if sys.platform == "win32":
    os.system("")

from retrieval.rag_search import search_information
from llm.chatbot import generate_answer_stream, rewrite_query_with_history

# =========================================================================
# ANSI COLOR SCHEME & STYLING
# =========================================================================
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
RED = "\033[91m"
GRAY = "\033[90m"
RESET = "\033[0m"

STANDARD_BENCHMARK_CONVERSATIONS = [
    "tomar naam ki",
    "kemon acho",
    "koyta dept ache",
    "BAUST এর ভিসি কে?",
    "eita english e bolo",
    "Who is the Head of CSE department?"
]


def execute_turn(q: str, history: list) -> str:
    """Executes a single conversational turn with live token streaming and latency metrics."""
    print(f"\n{BOLD}{CYAN}👤 USER:{RESET} {q}")
    
    # 1. Measure Query Rewriting Latency
    t0 = time.time()
    effective_q = rewrite_query_with_history(q, history)
    rewrite_time = time.time() - t0

    if effective_q != q:
        print(f"{GRAY}   ↳ Context Query Rewritten: \"{effective_q}\" ({rewrite_time:.3f}s){RESET}")

    # 2. Measure Hybrid RAG Search Latency
    t1 = time.time()
    ctx = search_information(effective_q)
    search_time = time.time() - t1

    print(f"{DIM}{YELLOW}🔍 [RAG Search: {search_time:.3f}s]{RESET}")
    print(f"{BOLD}{GREEN}🤖 AI RESPONSE:{RESET} ", end="", flush=True)

    # 3. Stream Tokens Live with TTFT (Time To First Token) Measurement
    gen_start = time.time()
    first_token_time = None
    ans_parts = []
    token_count = 0

    for token in generate_answer_stream(ctx, q, history=history):
        if first_token_time is None:
            first_token_time = time.time() - gen_start
        print(token, end="", flush=True)
        ans_parts.append(token)
        token_count += 1

    gen_time = time.time() - gen_start
    if first_token_time is None:
        first_token_time = gen_time

    tok_per_sec = (token_count / gen_time) if gen_time > 0 else 0
    full_ans = "".join(ans_parts).strip()

    # 4. Latency & Performance Telemetry
    print(
        f"\n{GRAY}⏱️  Latency Metrics: TTFT: {first_token_time:.3f}s | "
        f"Gen: {gen_time:.3f}s | Speed: {tok_per_sec:.1f} tok/s | "
        f"Total Turnaround: {(search_time + gen_time):.3f}s{RESET}\n"
    )

    # 5. Update Conversation Memory
    history.append({"role": "user", "content": q})
    history.append({"role": "assistant", "content": full_ans})
    return full_ans


def run_benchmark():
    """Runs the standard 6-turn automated test suite."""
    print(f"\n{BOLD}{MAGENTA}============================================================")
    print("   RUNNING AUTOMATED 6-TURN BENCHMARK EVALUATION")
    print(f"============================================================{RESET}")
    bench_history = []
    total_start = time.time()

    for i, q in enumerate(STANDARD_BENCHMARK_CONVERSATIONS, 1):
        print(f"{BOLD}{YELLOW}[Turn {i}/{len(STANDARD_BENCHMARK_CONVERSATIONS)}]{RESET}", end=" ")
        execute_turn(q, bench_history)

    total_duration = time.time() - total_start
    print(f"{BOLD}{GREEN}✅ BENCHMARK EVALUATION COMPLETE in {total_duration:.2f}s!{RESET}\n")


def print_banner():
    """Renders the CLI startup header."""
    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║            BAUST AI HELPDESK - INTERACTIVE CLI TESTER        ║{RESET}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════════╝{RESET}")
    print(f"{DIM}Real-time RAG Search • Streaming Generator • Latency Profiler{RESET}")
    print(f"{GRAY}Commands: 'exit'/'quit' to stop | 'clear' to reset history | 'history' to view log | 'benchmark' to run test-suite{RESET}\n")


def main():
    # Detect CLI flags: --auto or --test runs benchmark non-interactively
    auto_mode = any(arg in sys.argv for arg in ["--auto", "--test", "-t", "-a"])
    
    # If piped or explicitly requested, run automated benchmark
    if auto_mode or not sys.stdin.isatty():
        run_benchmark()
        return

    # Interactive Live Terminal Chat Mode
    print_banner()
    conversation_history = []

    while True:
        try:
            prompt_label = f"{BOLD}{BLUE}USER > {RESET}"
            user_input = input(prompt_label).strip()

            if not user_input:
                continue

            cmd_lower = user_input.lower()
            if cmd_lower in ["exit", "quit", "q"]:
                print(f"{YELLOW}Exiting interactive test session. Goodbye!{RESET}")
                break

            elif cmd_lower == "clear":
                conversation_history.clear()
                print(f"{GREEN}🧹 Conversation history and context memory cleared.{RESET}\n")
                continue

            elif cmd_lower == "history":
                if not conversation_history:
                    print(f"{GRAY}[History is currently empty]{RESET}\n")
                else:
                    print(f"\n{BOLD}{YELLOW}--- CONVERSATION HISTORY ({len(conversation_history) // 2} turns) ---{RESET}")
                    for idx, msg in enumerate(conversation_history, 1):
                        role_tag = f"{CYAN}User" if msg["role"] == "user" else f"{GREEN}Assistant"
                        print(f" {idx}. {role_tag}: {RESET}{msg['content'][:100]}...")
                    print()
                continue

            elif cmd_lower in ["benchmark", "auto", "test"]:
                run_benchmark()
                continue

            # Execute real-time streaming question
            execute_turn(user_input, conversation_history)

        except (KeyboardInterrupt, EOFError):
            print(f"\n{YELLOW}Session interrupted. Exiting...{RESET}")
            break
        except Exception as err:
            print(f"\n{RED}❌ Error during execution: {err}{RESET}\n")


if __name__ == "__main__":
    main()
