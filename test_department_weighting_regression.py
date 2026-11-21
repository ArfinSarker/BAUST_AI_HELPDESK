import unittest
import os
import sys
import re

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval.rag_search import get_store, search_information, reload_vector_store

class TestDepartmentWeightingRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Force a fresh load of the store to test in-memory normalization
        reload_vector_store()

    def test_01_metadata_key_parity(self):
        """Verify that all chunks contain both 'source' and 'filename' with identical provenance values."""
        docs, metadata, embeddings, faiss_idx = get_store()
        self.assertTrue(len(metadata) > 0, "Metadata store is empty!")
        
        for idx, meta in enumerate(metadata):
            self.assertIn("source", meta, f"Chunk {idx} missing 'source' key")
            self.assertIn("filename", meta, f"Chunk {idx} missing 'filename' key")
            self.assertEqual(
                meta["source"], 
                meta["filename"], 
                f"Chunk {idx} has mismatched provenance keys: source='{meta.get('source')}' vs filename='{meta.get('filename')}'"
            )
            self.assertTrue(len(meta["source"]) > 0, f"Chunk {idx} has empty provenance value")

    def test_02_cse_department_weighting(self):
        """Verify that queries mentioning CSE retrieve and prioritize CSE department files."""
        ctx = search_information("CSE department faculty members and lecturers")
        self.assertIn("cse_department_and_faculty.txt", ctx)
        self.assertNotIn("eee_department_and_faculty.txt", ctx)
        self.assertNotIn("ce_department_and_faculty.txt", ctx)

    def test_03_eee_department_weighting(self):
        """Verify that queries mentioning EEE retrieve and prioritize EEE department files."""
        ctx = search_information("EEE electrical engineering teachers and faculty directory")
        self.assertIn("eee_department_and_faculty.txt", ctx)
        self.assertNotIn("cse_department_and_faculty.txt", ctx)
        self.assertNotIn("me_department_and_faculty.txt", ctx)

    def test_04_ce_department_weighting(self):
        """Verify that queries mentioning Civil Engineering (CE) retrieve CE department files."""
        ctx = search_information("Civil engineering CE department teachers and faculty list")
        self.assertIn("ce_department_and_faculty.txt", ctx)
        self.assertNotIn("eee_department_and_faculty.txt", ctx)

    def test_05_me_department_weighting(self):
        """Verify that queries mentioning Mechanical Engineering (ME) retrieve ME department files."""
        ctx = search_information("Mechanical engineering ME department faculty details")
        self.assertIn("me_department_and_faculty.txt", ctx)
        self.assertNotIn("cse_department_and_faculty.txt", ctx)

    def test_06_ipe_department_weighting(self):
        """Verify that queries mentioning IPE retrieve IPE department files."""
        ctx = search_information("Industrial and production engineering IPE department info")
        self.assertIn("ipe_department_and_faculty.txt", ctx)
        self.assertNotIn("eee_department_and_faculty.txt", ctx)

    def test_07_bba_fbs_department_weighting(self):
        """Verify that queries mentioning BBA correctly map to and retrieve FBS department files."""
        ctx = search_information("BBA faculty of business studies teachers")
        self.assertIn("fbs_department_and_faculty.txt", ctx)
        self.assertNotIn("ce_department_and_faculty.txt", ctx)

    def test_08_cross_department_suppression(self):
        """Verify that cross-department files are heavily penalized and do not leak into results."""
        ctx = search_information("Computer science CSE laboratory facilities and teachers")
        # Ensure other department files are suppressed
        for other_doc in ["me_department_and_faculty.txt", "ce_department_and_faculty.txt", "eee_department_and_faculty.txt"]:
            self.assertNotIn(other_doc, ctx, f"Unrelated department file '{other_doc}' leaked into CSE query context!")

    def test_09_general_institutional_files_retention(self):
        """Verify that general institutional files (academic calendar, admission) are preserved and boosted properly."""
        ctx_cal = search_information("When are the summer vacation holidays in academic calendar?")
        self.assertIn("academic_calendar.txt", ctx_cal)

        ctx_adm = search_information("What are the admission requirements and tuition fees?")
        self.assertTrue(
            "admission_information.txt" in ctx_adm or "tuition_fees_and_scholarships.txt" in ctx_adm,
            "Failed to retrieve general admission or tuition documentation!"
        )

    def test_10_exhaustive_multi_department_isolation_matrix(self):
        """Matrix test: For each department, verify target files are retrieved and all competing department files are isolated."""
        dept_file_map = {
            "cse": "cse_department_and_faculty.txt",
            "eee": "eee_department_and_faculty.txt",
            "ce": "ce_department_and_faculty.txt",
            "me": "me_department_and_faculty.txt",
            "ipe": "ipe_department_and_faculty.txt",
            "bba": "fbs_department_and_faculty.txt"
        }

        queries = {
            "cse": "CSE department professors, lecturers and course offerings",
            "eee": "EEE department electrical engineering lab faculty members",
            "ce": "Civil engineering CE department faculty list and officers",
            "me": "Mechanical engineering ME department faculty members",
            "ipe": "Industrial and Production Engineering IPE department teachers",
            "bba": "BBA Faculty of Business Studies FBS department lecturers"
        }

        for target_dept, target_file in dept_file_map.items():
            query = queries[target_dept]
            ctx = search_information(query, top_k=4)

            # Assert target department file is present
            self.assertIn(
                target_file, 
                ctx, 
                f"Target file '{target_file}' was not retrieved for query: '{query}'"
            )

            # Assert no competing department files leaked into the top-k results
            competing_files = [f for d, f in dept_file_map.items() if d != target_dept]
            for comp_file in competing_files:
                self.assertNotIn(
                    comp_file, 
                    ctx, 
                    f"Department isolation violated: '{comp_file}' leaked into results for '{target_dept}' query!"
                )

    def test_11_ambiguous_common_phrasing_isolation(self):
        """Verify that when queries use identical generic phrasing, the department token enforces strict isolation."""
        common_phrase = "Who is the head of the department and faculty members?"
        
        # Test CSE isolation
        ctx_cse = search_information(f"CSE: {common_phrase}", top_k=4)
        self.assertIn("cse_department_and_faculty.txt", ctx_cse)
        self.assertNotIn("eee_department_and_faculty.txt", ctx_cse)
        self.assertNotIn("me_department_and_faculty.txt", ctx_cse)
        self.assertNotIn("ce_department_and_faculty.txt", ctx_cse)

        # Test EEE isolation
        ctx_eee = search_information(f"EEE: {common_phrase}", top_k=4)
        self.assertIn("eee_department_and_faculty.txt", ctx_eee)
        self.assertNotIn("cse_department_and_faculty.txt", ctx_eee)
        self.assertNotIn("me_department_and_faculty.txt", ctx_eee)
        self.assertNotIn("ce_department_and_faculty.txt", ctx_eee)

        # Test CE isolation
        ctx_ce = search_information(f"CE: {common_phrase}", top_k=4)
        self.assertIn("ce_department_and_faculty.txt", ctx_ce)
        self.assertNotIn("cse_department_and_faculty.txt", ctx_ce)
        self.assertNotIn("eee_department_and_faculty.txt", ctx_ce)

        # Test ME isolation
        ctx_me = search_information(f"ME: {common_phrase}", top_k=4)
        self.assertIn("me_department_and_faculty.txt", ctx_me)
        self.assertNotIn("cse_department_and_faculty.txt", ctx_me)
        self.assertNotIn("eee_department_and_faculty.txt", ctx_me)

    def test_12_metadata_provenance_field_normalization_contract(self):
        """Ensure contract integrity: indexing outputs and store outputs normalize 'source' and 'filename' interchangeably."""
        docs, metadata, embeddings, faiss_idx = get_store()
        for m in metadata:
            # Code reading .get('source') and code reading .get('filename') must receive identical provenance
            self.assertEqual(m.get('source'), m.get('filename'))
            self.assertIsNotNone(m.get('source'))
            self.assertIsNotNone(m.get('filename'))

if __name__ == "__main__":
    unittest.main()
