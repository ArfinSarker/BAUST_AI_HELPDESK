import unittest
import time
import threading
from metadata_manager import (
    get_db_connection_safe,
    infer_metadata_from_text,
    seed_metadata_async,
    get_all_doc_types,
    get_all_authorities,
    get_document_metadata,
    get_all_documents_metadata,
    register_or_update_document
)


class TestMetadataManagerUpgrades(unittest.TestCase):

    def test_01_context_manager_safety(self):
        """Verify get_db_connection_safe opens WAL connection and closes reliably."""
        with get_db_connection_safe() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode;")
            mode = cursor.fetchone()[0]
            self.assertEqual(mode.lower(), "wal")

        # Verify rollback on exception
        with self.assertRaises(ValueError):
            with get_db_connection_safe() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1;")
                raise ValueError("Simulated failure inside transaction")

    def test_02_regex_backtracking_and_dos_protection(self):
        """Verify regex NLP does not hang or suffer catastrophic backtracking on corrupt or pathological inputs."""
        # 1. Pathological whitespace test (50,000 spaces & symbols)
        pathological_input = "office of the " + (" " * 50000) + "proctor baust"
        start_time = time.time()
        meta = infer_metadata_from_text("test_doc.txt", pathological_input)
        duration = time.time() - start_time

        self.assertLess(duration, 0.2, f"Regex took too long ({duration:.3f}s), possible catastrophic backtracking!")
        self.assertIsNotNone(meta["title"])

        # 2. Corrupt / binary-like garbled text
        garbled = "department of the \x00\x01\x02 " * 2000
        start_time = time.time()
        meta2 = infer_metadata_from_text("garbled.txt", garbled)
        duration2 = time.time() - start_time
        self.assertLess(duration2, 0.2, f"Regex garbled text took too long ({duration2:.3f}s)!")

        # 3. Legitimate extraction test
        sample = """
        Office of the Registrar
        Ref: BAUST/REG/NOTICE/2025/08
        Date: 15 October 2025
        Subject: Class Routine for Fall 2025 Semester
        This is to notify all students...
        """
        meta3 = infer_metadata_from_text("class_routine_fall_2025.txt", sample)
        self.assertEqual(meta3["academic_year"], "2025")
        self.assertEqual(meta3["semester"], "Fall")
        self.assertEqual(meta3["doc_type"], "routine")
        self.assertEqual(meta3["issuing_authority"], "Office of the Registrar")
        self.assertIn("Class Routine", meta3["title"])

    def test_03_concurrent_lock_free_reads(self):
        """Verify multiple threads can concurrently execute read queries without deadlocks."""
        errors = []

        def reader_task():
            try:
                for _ in range(10):
                    doc_types = get_all_doc_types()
                    authorities = get_all_authorities()
                    all_meta = get_all_documents_metadata()
                    self.assertIsInstance(doc_types, list)
                    self.assertIsInstance(authorities, list)
                    self.assertIsInstance(all_meta, dict)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader_task) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        self.assertEqual(len(errors), 0, f"Concurrent reads produced errors: {errors}")

    def test_04_background_async_seeding(self):
        """Verify seed_metadata_async launches worker thread without blocking caller."""
        start_time = time.time()
        thread = seed_metadata_async()
        spawn_time = time.time() - start_time

        # Seeding invocation must return in less than 50ms
        self.assertLess(spawn_time, 0.1, "seed_metadata_async took too long to return!")
        self.assertIsInstance(thread, threading.Thread)
        thread.join(timeout=5.0)


if __name__ == "__main__":
    unittest.main()

