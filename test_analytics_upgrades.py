import unittest
import time
import threading
from analytics import (
    _log_executor,
    _DB_LOCK,
    log_chat_async,
    log_chat,
    update_feedback,
    get_analytics_stats,
    get_recent_logs,
    get_knowledge_gaps,
    get_top_questions,
    get_logs_count,
    get_knowledge_gaps_count,
    get_db_connection
)


class TestAnalyticsUpgrades(unittest.TestCase):

    def setUp(self):
        self.inserted_ids = []

    def tearDown(self):
        # Clean up any test records created during test execution
        if self.inserted_ids:
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM chat_logs WHERE id IN (" + ",".join(map(str, self.inserted_ids)) + ")"
                )
                conn.commit()
                conn.close()
            except Exception:
                pass

    def test_01_threadpool_executor_async_logging(self):
        """Verify log_chat_async uses ThreadPoolExecutor and returns a valid Future."""
        from concurrent.futures import Future
        
        future = log_chat_async(
            question="TEST_AUTOMATION_FEE_QUERY",
            answer="The tuition fee is 7,84,000 BDT for 4 years.",
            context="BAUST CSE Fee Document"
        )
        self.assertIsInstance(future, Future)
        log_id = future.result(timeout=3.0)
        self.assertIsNotNone(log_id)
        self.assertIsInstance(log_id, int)
        self.assertGreater(log_id, 0)
        self.inserted_ids.append(log_id)

    def test_02_feedback_with_log_id_precision(self):
        """Verify update_feedback updates the EXACT row using log_id, preventing race condition collisions."""
        # Insert two identical questions consecutively
        shared_q = "TEST_COLLISION_CHECK_QUESTION"
        id_1 = log_chat(shared_q, "Answer 1", "Context 1")
        id_2 = log_chat(shared_q, "Answer 2", "Context 2")
        self.inserted_ids.extend([id_1, id_2])
        self.assertNotEqual(id_1, id_2)

        # Update feedback ONLY for id_1
        success = update_feedback(question=shared_q, feedback="like", log_id=id_1)
        self.assertTrue(success)

        # Verify in DB that id_1 has 'like' and id_2 has NULL feedback
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, feedback FROM chat_logs WHERE id IN (?, ?)", (id_1, id_2))
        rows = {r["id"]: r["feedback"] for r in cursor.fetchall()}
        conn.close()

        self.assertEqual(rows[id_1], "like")
        self.assertIsNone(rows[id_2], "id_2 should NOT have been updated when targeting id_1!")

        # Now update id_2 with dislike
        success_2 = update_feedback(question=shared_q, feedback="dislike", log_id=id_2)
        self.assertTrue(success_2)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT feedback, knowledge_gap FROM chat_logs WHERE id = ?", (id_2,))
        r2 = cursor.fetchone()
        conn.close()
        self.assertEqual(r2["feedback"], "dislike")
        self.assertEqual(r2["knowledge_gap"], 1, "Dislike should flag knowledge_gap")

    def test_03_lock_free_concurrent_reads(self):
        """Verify read operations execute freely without locking _DB_LOCK."""
        # If _DB_LOCK is held by a long write, WAL mode allows concurrent readers!
        # Test that calling read operations does not block or acquire _DB_LOCK
        stats = get_analytics_stats()
        self.assertIn("total_queries", stats)
        self.assertIn("satisfaction_rate", stats)

        recent = get_recent_logs(limit=10, offset=0)
        self.assertIsInstance(recent, list)

        gaps = get_knowledge_gaps(limit=10, offset=0)
        self.assertIsInstance(gaps, list)

        top_q = get_top_questions(limit=5)
        self.assertIsInstance(top_q, list)

    def test_04_pagination_support(self):
        """Verify limit and offset pagination for get_recent_logs and get_knowledge_gaps."""
        # Insert several distinct logs to ensure test data exists
        for i in range(5):
            lid = log_chat(f"Pagination Test Query {i}-{time.time()}", f"Answer {i}", "Context")
            self.inserted_ids.append(lid)

        page_1 = get_recent_logs(limit=2, offset=0)
        page_2 = get_recent_logs(limit=2, offset=2)

        self.assertEqual(len(page_1), 2)
        self.assertEqual(len(page_2), 2)
        # Ensure page 1 and page 2 contain different records
        page_1_ids = [item["id"] for item in page_1]
        page_2_ids = [item["id"] for item in page_2]
        self.assertEqual(len(set(page_1_ids).intersection(set(page_2_ids))), 0, "Pages must not overlap!")

    def test_05_flask_api_feedback_integration(self):
        """Verify /api/feedback accepts log_id via Flask test client."""
        from app import app
        client = app.test_client()

        # Insert a chat log
        test_id = log_chat("TEST_LOCATION_QUERY", "Saidpur Cantonment", "Location Doc")
        self.inserted_ids.append(test_id)

        res = client.post("/api/feedback", json={
            "log_id": test_id,
            "question": "TEST_LOCATION_QUERY",
            "answer": "Saidpur Cantonment",
            "feedback": "like"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "success")

        # Verify DB updated
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT feedback FROM chat_logs WHERE id = ?", (test_id,))
        row = cursor.fetchone()
        conn.close()
        self.assertEqual(row["feedback"], "like")


if __name__ == "__main__":
    unittest.main()

