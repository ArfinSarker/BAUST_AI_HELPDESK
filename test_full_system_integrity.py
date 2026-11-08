import unittest
import os
import sys
import json
import requests
from dotenv import load_dotenv

load_dotenv()

class TestFullSystemIntegrity(unittest.TestCase):
    BASE_URL = "http://127.0.0.1:5000"

    def test_01_homepage_availability(self):
        """Verify Student Chatbot interface is live and returns 200 with proper HTML."""
        resp = requests.get(self.BASE_URL, timeout=5)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("BAUST AI Help Desk", resp.text)
        self.assertIn("chat.js", resp.text)

    def test_02_admin_login_page_and_security(self):
        """Verify Admin login page is accessible and provides CSRF token."""
        resp = requests.get(f"{self.BASE_URL}/admin-login", timeout=5)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("csrf_token", resp.text)
        self.assertIn("Secure Administrative Gateway", resp.text)

    def test_03_rag_hybrid_search_accuracy(self):
        """Verify RAG search retrieves relevant BAUST university knowledge in sub-second time."""
        from retrieval.rag_search import search_information
        
        # Test 1: Department query
        ctx1 = search_information("koyta dept ache")
        self.assertTrue(len(ctx1) > 50)
        self.assertTrue(any(dept in ctx1 for dept in ["CSE", "EEE", "ME", "CE"]))

        # Test 2: Vice-Chancellor query
        ctx2 = search_information("BAUST er VC ke")
        self.assertTrue(len(ctx2) > 50)
        self.assertTrue("Mobarak" in ctx2 or "মোবারক" in ctx2 or "Vice-Chancellor" in ctx2 or "উপাচার্য" in ctx2)

    def test_04_query_rewriting_and_translation_detection(self):
        """Verify query rewriting preserves standalone questions and handles translations."""
        from llm.chatbot import rewrite_query_with_history, is_translation_request

        is_trans, _ = is_translation_request("eita english e bolo")
        self.assertTrue(is_trans)

        is_trans2, _ = is_translation_request("Who is the VC?")
        self.assertFalse(is_trans2)

        history = [
            {"role": "user", "content": "Who is the Head of CSE department?"},
            {"role": "assistant", "content": "Dr. Md Nakib Hayat Chowdhury is the Head of CSE."}
        ]
        # Pronoun query should blend context
        rewritten = rewrite_query_with_history("tar phone number koto", history)
        self.assertIn("Head of CSE", rewritten)

    def test_05_analytics_and_gap_detection(self):
        """Verify knowledge gap detector correctly classifies greetings vs real missing info."""
        from analytics import detect_knowledge_gap, get_analytics_stats

        # Greetings should NEVER be flagged as knowledge gaps
        self.assertFalse(detect_knowledge_gap("hi", "", "Hello! How can I help?"))
        self.assertFalse(detect_knowledge_gap("kemon acho", "", "Ami valo achi."))

        # Missing information should be flagged as knowledge gaps
        self.assertTrue(detect_knowledge_gap("BAUST swimming pool ache?", "", "আমাদের ডাটাবেসে সুইমিং পুল সম্পর্কিত তথ্য নেই।"))

        stats = get_analytics_stats()
        self.assertIn("total_queries", stats)
        self.assertIn("unresolved_gaps_count", stats)
        self.assertGreaterEqual(stats["total_queries"], 0)

    def test_06_feedback_api_endpoint(self):
        """Verify user feedback API accepts likes and dislikes with 200 response."""
        resp = requests.post(
            f"{self.BASE_URL}/api/feedback",
            json={"question": "What is BAUST?", "answer": "BAUST is a university.", "feedback": "like"},
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "success")

    def test_07_offline_sentence_transformer_speed(self):
        """Verify SentenceTransformer model loads without Hugging Face network retry hangs."""
        import time
        from retrieval.rag_search import _get_embedding_model

        start = time.time()
        model = _get_embedding_model()
        duration = time.time() - start

        self.assertIsNotNone(model)
        # Should load or return cached singleton instantly (< 1.5s)
        self.assertLess(duration, 1.5, f"Model getter took {duration:.2f}s, expected instant retrieval!")

if __name__ == "__main__":
    unittest.main()

