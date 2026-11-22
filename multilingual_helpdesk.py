"""
Production-Ready Multi-Lingual AI Helpdesk Backend Module
Built with the Official Google GenAI SDK (`google-genai`)
"""

import os
import logging
from typing import Generator, List, Dict, Any, Optional
from google.genai import client, types
from google.genai import errors

# Configure module logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MultilingualHelpdesk")


# =====================================================================
# SYSTEM INSTRUCTION FOR SMART LANGUAGE MAPPING & RAG GROUNDING
# =====================================================================
SYSTEM_INSTRUCTION = """You are an intelligent, authoritative, and helpful AI Helpdesk Assistant.

LANGUAGE MAPPING RULES (STRICTLY ENFORCE VIA SYSTEM INSTRUCTION):
1. **Bengali Input (বাংলা)**:
   - If the user writes in native Bengali script (বাংলা), reply 100% in pure, natural, and grammatically correct Bengali script.
2. **English Input**:
   - If the user writes in standard English, reply 100% in clear, professional English.
3. **Banglish / Romanized Bengali Input**:
   - If the user writes in Banglish / Romanized Bengali (e.g., "Aamar nam Imran", "Bhai, ki obostha?", "Admission fee koto?"):
   - You MUST reply in proper Bengali script (বাংলা বর্ণমালায়), NEVER in Romanized Banglish!
   - Example: For "class kobe shuru?", reply in Bengali: "ক্লাস আগামী ১৫ জানুয়ারি থেকে শুরু হবে।"
4. **Translation Requests**:
   - If the user explicitly asks to translate, convert, or summarize previous responses (e.g., "eta banglay translate kore dao", "in English please"):
   - Accurately translate or convert the preceding turn from conversation history into the requested target language.

STUDENT PROFILE & CLASS ROUTINE RULES:
1. **Class Routine & Schedule Requests**:
   - When a student asks about their class routine or today/tomorrow's classes:
   - Identify the student (Name, Roll, Department, Level-Term) at the top in clear bullet points.
   - Note weekend/holidays (Friday & Saturday are weekly holidays in Bangladesh; explain that there are no classes, and show the schedule for Sunday).
   - List the scheduled classes with time, course code and title, room/lab, and teacher.
   - LAB DURATION RULE (CRITICAL):
     * In BAUST routines, all Sessional Labs (e.g. CSE 3202, CSE 3204, CSE 3206, CSE 3210, CSE 3200) run for 3 contact hours (3 consecutive class periods):
       - Morning lab (Periods 1-3): 08:00 AM - 10:55 AM (3 hours)
       - Afternoon lab (Periods 4-6): 11:30 AM - 02:25 PM (3 hours)
       - Afternoon lab (Periods 7-9): 02:30 PM - 05:25 PM (3 hours)
     * Always display the full 3-hour duration for sessional labs as given in the verified schedule (e.g. '08:00 AM - 10:55 AM'). NEVER display a 3-hour sessional lab as only 55 minutes ('08:00 AM - 08:55 AM')!
   - LAB EVEN/ODD LABELLING & DURATION RULE (CRITICAL):
     * In BAUST routines, the labels '#EVEN#' and '#ODD#' for sessional labs represent alternating bi-weekly lab group designations:
       - For Section A: `CSE 3202` is an `#EVEN#` lab, and `CSE 3210` is an `#ODD#` lab.
       - For Section B: `CSE 3202` is an `#ODD#` lab, and `CSE 3210` is an `#EVEN#` lab.
     * Whenever a lab in the verified schedule has an `(#EVEN# ল্যাব)` or `(#ODD# ল্যাব)` label (e.g. `CSE 3202 (Artificial Intelligence Sessional) `(#EVEN# ল্যাব)``), you MUST wrap the label in backticks (e.g. ` `(#EVEN# ল্যাব)` ` or ` `(#ODD# ল্যাব)` `) so it renders as a prominent highlighted badge!
     * NEVER state that students are excluded from the class (all students of that section attend this section routine). Present the scheduled class with its Course Code, Title, highlighted `(#EVEN# / #ODD#)` badge, Room/Lab, and Teacher.
   - In BAUST CSE Level 3 Term 2, Sunday (Sun) is a REGULAR academic class day for both Section A and Section B. Present the exact classes from the authoritative routine for the requested section. NEVER state that Sunday is empty or has no classes!
   - NEVER substitute a class routine question with an exam seatplan unless exam seat was asked.

2. **Faculty Routine Short Form & Teacher Initials**:
   - In BAUST routines, teachers are indicated by short forms (e.g. `GR` = S. M Golam Rifat / এস. এম. গোলাম রিফাত, `FTZ` = Fatema Tuz Zohra, `AKZ` = A K Z Rasel Rahman, `MI` = Md. Minhazul Islam, `MZH` = Md Zahim Hassan, `RH` = MD. Rabby Hossain, `GC` = Gourab Chandra Biswas, `ASF` = Md. Asif Iqbal).
   - When asked who a teacher is by their short form (e.g. 'GR ke'):
     * NEVER hallucinate or guess any other person (such as Golam Rabbani).
     * Output their exact full name, designation, department, and routine courses.

3. **Saidpur Spelling Rule (CRITICAL)**:
   - Always spell Saidpur in Bengali as **সৈয়দপুর** বা **সৈয়দপুর** (e.g. "বাংলাদেশ আর্মি ইউনিভার্সিটি অব সায়েন্স অ্যান্ড টেকনোলজি (BAUST), সৈয়দপুর")।
   - NEVER write 'সাইদপুর' or 'সায়েদপুর'!

4. **Graduate / Masters & MBA Programs Rule (CRITICAL)**:
   - When asked whether Masters or MBA is available at BAUST ("masters kora jay?", "MBA ache?", "postgraduate program ki ki", "MBA kora jay?"):
   - Clearly confirm that **হ্যাঁ, BAUST সৈয়দপুরে মাস্টার্স (স্নাতকোত্তর) ও MBA করার পূর্ণ সুযোগ রয়েছে।**
   - বিশ্ববিদ্যালয়ের **ব্যবসায় প্রশাসন বিভাগ (Department of Business Administration - DBA)** / **Faculty of Business Studies (FBS)** থেকে **Master of Business Administration (MBA)** অফার করা হয়।
   - আপনি অবশ্যই সর্বদা নিচের ২টি প্রধান MBA ট্র্যাক বিস্তারিতভাবে উল্লেখ করবেন:
     1. **Regular MBA (রেগুলার এমবিএ):**
        - **BBA গ্র্যাজুয়েটদের জন্য:** ১ বছর মেয়াদি (২ সেমিস্টার, ৩৬ ক্রেডিট)।
        - **অন্যান্য নন-বিজনেস ব্যাকগ্রাউন্ড (B.Sc, BA, BSS ইত্যাদি) গ্র্যাজুয়েটদের জন্য:** ২ বছর মেয়াদি (৪ সেমিস্টার, ৬০ ক্রেডিট)।
     2. **Executive MBA / EMBA (এক্সিকিউটিভ এমবিএ):**
        - চাকুরিজীবী বা কর্মরত পেশাজীবীদের (Working Professionals) জন্য বিশেষ সুবিধাজনক সময়ে (উইকেন্ড/সান্ধ্যকালীন ক্লাস) পরিচালিত হয়। মেয়াদ ১.৫ বছর (৩ সেমিস্টার, ৪৮ ক্রেডিট)।
   - **MBA প্রোগ্রাম কো-অর্ডিনেটর:** মো. জাবের হোসেন (সহযোগী অধ্যাপক, DBA), মোবাইল: [01922626111](tel:01922626111), ইমেইল: [hossain.zaber478@gmail.com](mailto:hossain.zaber478@gmail.com)।
   - ভর্তি অফিসের হটলাইন: [01769675580](tel:01769675580), [01769675588](tel:01769675588)।
   - এছাড়াও ইংরেজি বিভাগ থেকে **MA in English (ELT)** (১ বছর মেয়াদি, ৩৬ ক্রেডিট) পরিচালিত হয়।
   - **সতর্কতা (DO NOT HALLUCINATE):** BAUST-এর ইঞ্জিনিয়ারিং বিভাগসমূহে (CSE, EEE, ME, IPE, CE) বর্তমানে শুধুমাত্র ৪ বছর মেয়াদি আন্ডারগ্র্যাজুয়েট B.Sc. Engineering চালু রয়েছে; ইঞ্জিনিয়ারিংয়ে কোনো M.Sc. বা মাস্টার্স চালু নেই। ইঞ্জিনিয়ারিংয়ে মাস্টার্স আছে বলে কখনোই উল্লেখ করবেন না।

RAG GROUNDING & INTEGRITY RULES (ANTI-HALLUCINATION):
1. **Context-Bounded Answers**:
   - You are provided with a dedicated "RETRIEVED KNOWLEDGE BASE CONTEXT".
   - You MUST base your factual answers STRICTLY on this provided context.
   - Do NOT assume, extrapolate, or hallucinate facts that are not explicitly present in the context.
2. **Missing Information Protocol**:
   - If the provided context does not contain the answer to the user's question, politely state in the appropriate language that the information is currently not available in the database, and advise contacting official support.
3. **Formatting & Contact Links (MANDATORY)**:
   - Provide crisp, structured responses using markdown bullet points for readability. Wrap key identifiers (codes, dates, fees, room numbers) in backticks.
   - For phone and email contacts, format each as an individual clickable link:
     e.g., `* **মোবাইল / ফোন (Phone):** [01759810833](tel:01759810833) (WhatsApp)`
     e.g., `* **ইমেইল (Email):** [email1@baust.edu.bd](mailto:email1@baust.edu.bd), [email2@gmail.com](mailto:email2@gmail.com)`
   - NEVER combine multiple emails into a single link like `[a, b](mailto:a, b)`. Each email and phone must be a separate link.
   - For educational qualifications, format cleanly (e.g. `B.Sc. in CSE (KU), M.Sc. in CSE (KU)` or `বিএসসি ইন সিএসই (খুলনা বিশ্ববিদ্যালয় - KU)`). NEVER write repetitive or broken translations like `বি.এসসি (বি.এসসি. ইন সি.এসই, কেইউ)`.
   - For phone and email contacts, format each as an individual clickable link:
     e.g., `* **মোবাইল / ফোন (Phone):** [01759810833](tel:01759810833) (WhatsApp)`
     e.g., `* **ইমেইল (Email):** [email1@baust.edu.bd](mailto:email1@baust.edu.bd), [email2@gmail.com](mailto:email2@gmail.com)`
   - NEVER combine multiple emails into a single link like `[a, b](mailto:a, b)`. Each email and phone must be a separate link.
   - For educational qualifications, format cleanly (e.g. `B.Sc. in CSE (KU), M.Sc. in CSE (KU)` or `বিএসসি ইন সিএসই (খুলনা বিশ্ববিদ্যালয় - KU)`). NEVER write repetitive or broken translations like `বি.এসসি (বি.এসসি. ইন সি.এসই, কেইউ)`.
"""


def trim_history(
    history: Optional[List[Dict[str, str]]],
    max_messages: int = 4,
    current_query: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Token-efficient Sliding Window History trimmer with Duplication Prevention.
    
    Accepts the conversation history array and slices it to retain only the last
    `max_messages` (default 4 messages / 2 conversation turns).
    
    Prevents duplicate submissions if the frontend mistakenly already appended
    the current query to the history array prior to making the API call.
    
    Args:
        history: List of message dicts with keys 'role' and 'content'/'text'.
        max_messages: Maximum recent messages to preserve.
        current_query: The incoming query to check for and deduplicate against.
        
    Returns:
        List of deduplicated, trimmed message dicts.
    """
    if not history:
        return []
        
    clean_history = list(history)
    
    # History Duplication Prevention: strip last item if it matches the current user query
    if current_query and clean_history:
        last_msg = clean_history[-1]
        last_text = (last_msg.get("content") or last_msg.get("text") or "").strip()
        last_role = last_msg.get("role", "").lower()
        if last_role in ["user", "human"] and last_text == current_query.strip():
            clean_history.pop()

    return clean_history[-max_messages:]


class MultilingualHelpdesk:
    """
    Production-grade AI Helpdesk backend orchestrator using the official Google GenAI SDK.
    Supports smart language mapping, RAG anti-hallucination, sliding window history,
    safety optimizations, synchronous inference, and SSE token streaming.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        fallback_model: str = "gemini-2.0-flash-lite-preview-02-05",
        temperature: float = 0.2
    ):
        """
        Initialize the Multilingual Helpdesk client.
        
        Args:
            api_key: Gemini API key. Defaults to GEMINI_API_KEY environment variable.
            model: Primary Gemini model (default: 'gemini-2.5-flash').
            fallback_model: Secondary model in case primary encounters quota/version limits.
            temperature: Sampling temperature for deterministic factual answers.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be provided or set in environment variables.")

        # Initialize official Google GenAI Client
        self.client = client.Client(api_key=self.api_key)
        self.model = model
        self.fallback_model = fallback_model
        self.temperature = temperature

        # Build optimized GenerateContentConfig
        # Includes safety setting optimization for university context (blood donation, medical fees)
        self._config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=self.temperature,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH"
                )
            ]
        )

    def _build_contents(
        self,
        query: str,
        context: str = "",
        history: Optional[List[Dict[str, str]]] = None,
        max_history_messages: int = 4
    ) -> List[types.Content]:
        """Construct structured multi-turn Content objects with RAG context grounding."""
        contents: List[types.Content] = []

        # 1. Process trimmed conversation history (Sliding Window with Duplication Check)
        trimmed = trim_history(
            history,
            max_messages=max_history_messages,
            current_query=query
        )
        for msg in trimmed:
            role = msg.get("role", "user")
            # Normalize frontend role names to Gemini API roles ('user' | 'model')
            if role in ["assistant", "bot", "model"]:
                role = "model"
            else:
                role = "user"

            text = msg.get("content") or msg.get("text") or ""
            if text.strip():
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=text.strip())]
                    )
                )

        # 2. Package current query with RAG context (Grounding injection)
        user_prompt_parts = []
        if context and context.strip():
            user_prompt_parts.append(
                f"### RETRIEVED KNOWLEDGE BASE CONTEXT (STRICT TRUTH SOURCE):\n{context.strip()}\n"
            )

        user_prompt_parts.append(f"### USER QUERY:\n{query.strip()}")

        current_user_turn = types.Content(
            role="user",
            parts=[types.Part.from_text(text="\n".join(user_prompt_parts))]
        )
        contents.append(current_user_turn)

        return contents

    def _get_candidate_models(self) -> List[str]:
        """Returns list of candidate models with resilient fallbacks."""
        # Include primary, requested fallback, and latest stable preview
        candidates = [self.model, self.fallback_model, "gemini-3.5-flash-lite"]
        # Remove duplicates while preserving priority order
        return [m for m in dict.fromkeys(candidates) if m]

    def generate_response(
        self,
        query: str,
        context: str = "",
        history: Optional[List[Dict[str, str]]] = None,
        max_history_messages: int = 4
    ) -> str:
        """
        Synchronous standard inference method.
        
        Args:
            query: User's query (Bengali, English, or Banglish).
            context: Retrieved RAG context.
            history: Previous conversation messages.
            max_history_messages: Max messages to retain in context window.
            
        Returns:
            Clean text response adhering strictly to language and RAG rules.
        """
        if not query or not query.strip():
            return ""

        contents = self._build_contents(
            query=query,
            context=context,
            history=history,
            max_history_messages=max_history_messages
        )

        candidate_models = self._get_candidate_models()

        for model_id in candidate_models:
            try:
                response = self.client.models.generate_content(
                    model=model_id,
                    contents=contents,
                    config=self._config
                )
                if response and response.text:
                    return response.text.strip()
            except errors.ClientError as ce:
                # Handle 404/NOT_FOUND or retired versions gracefully
                if "404" in str(ce) or "NOT_FOUND" in str(ce):
                    logger.warning(f"Model '{model_id}' unavailable (404). Falling back...")
                    continue
                logger.error(f"GenAI Client Error ({model_id}): {ce}")
                raise
            except Exception as e:
                logger.error(f"Inference error with model '{model_id}': {e}", exc_info=True)
                if model_id == candidate_models[-1]:
                    raise

        return ""

    def generate_response_stream(
        self,
        query: str,
        context: str = "",
        history: Optional[List[Dict[str, str]]] = None,
        max_history_messages: int = 4
    ) -> Generator[str, None, None]:
        """
        Real-Time Server-Sent Events (SSE) streaming generator.
        Yields text token chunks for real-time typing effect.
        
        Args:
            query: User's query (Bengali, English, or Banglish).
            context: Retrieved RAG context.
            history: Previous conversation messages.
            max_history_messages: Max messages to retain in context window.
            
        Yields:
            Token text chunks suitable for HTTP SSE streaming.
        """
        if not query or not query.strip():
            return

        contents = self._build_contents(
            query=query,
            context=context,
            history=history,
            max_history_messages=max_history_messages
        )

        candidate_models = self._get_candidate_models()

        for model_id in candidate_models:
            try:
                stream = self.client.models.generate_content_stream(
                    model=model_id,
                    contents=contents,
                    config=self._config
                )
                for chunk in stream:
                    if chunk.text:
                        yield chunk.text
                return  # Stream completed successfully
            except errors.ClientError as ce:
                if "404" in str(ce) or "NOT_FOUND" in str(ce):
                    logger.warning(f"Streaming: model '{model_id}' unavailable (404). Falling back...")
                    continue
                logger.error(f"GenAI Streaming Client Error ({model_id}): {ce}")
                raise
            except Exception as e:
                logger.error(f"Streaming error with model '{model_id}': {e}", exc_info=True)
                if model_id == candidate_models[-1]:
                    raise


# =====================================================================
# EXAMPLE USAGE & VERIFICATION
# =====================================================================
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # 1. Initialize MultilingualHelpdesk
    helpdesk = MultilingualHelpdesk()

    # 2. Mock RAG Context (from FAISS/BM25)
    sample_context = """
    Notice: BAUST Admission Notice for Summer 2026
    - Department: Computer Science and Engineering (CSE)
    - Admission Fee: 65,000 BDT
    - Per Semester Fee: 90,000 BDT
    - Total 4-Year Tuition: 6,95,000 BDT
    - Class Start Date: October 20, 2026
    - Helpline: 01769675588
    """

    print("=" * 70)
    print("DEMO: Streaming Banglish Query with RAG Grounding")
    print("=" * 70)

    banglish_query = "Bhai, CSE admission fee koto r class kobe shuru?"
    print(f"User Query (Banglish): '{banglish_query}'\n")
    print("AI Response (Bengali Script): ", end="", flush=True)

    # Stream real-time token chunks
    for token in helpdesk.generate_response_stream(
        query=banglish_query,
        context=sample_context
    ):
        print(token, end="", flush=True)

    print("\n" + "=" * 70)
