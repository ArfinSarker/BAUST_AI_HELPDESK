import os
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_KNOWLEDGE_FOLDER = os.path.join(BASE_DIR, "knowledge_base")
DEFAULT_VECTOR_FOLDER = os.path.join(BASE_DIR, "vector_store")
DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

_model_instance = None


def get_embedding_model(model_name=DEFAULT_MODEL_NAME):
    """Singleton getter for the SentenceTransformer model with lazy loading."""
    global _model_instance
    if _model_instance is None:
        print(f"[Embedding] Loading embedding model: {model_name}...")
        from sentence_transformers import SentenceTransformer
        _model_instance = SentenceTransformer(model_name)
    return _model_instance


def chunk_text(text, chunk_size=350, overlap=80):
    """
    Chunk text while preserving line breaks and structural formatting.
    """
    if not text or not text.strip():
        return []

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    step = max(1, chunk_size - overlap)
    chunks = []
    current_chunk_lines = []
    current_word_count = 0

    for line in lines:
        line_word_count = len(line.split())

        # Split extraordinarily long single lines
        if line_word_count > chunk_size:
            words = line.split()
            start = 0
            while start < len(words):
                end = start + chunk_size
                chunk_slice = " ".join(words[start:end])
                chunks.append(chunk_slice)
                start += step
            continue

        if current_word_count + line_word_count > chunk_size and current_chunk_lines:
            chunks.append("\n".join(current_chunk_lines))
            
            # Retain lines for overlap
            overlap_lines = []
            overlap_count = 0
            for prev_line in reversed(current_chunk_lines):
                prev_count = len(prev_line.split())
                if overlap_count + prev_count <= overlap:
                    overlap_lines.insert(0, prev_line)
                    overlap_count += prev_count
                else:
                    break
            current_chunk_lines = overlap_lines
            current_word_count = overlap_count

        current_chunk_lines.append(line)
        current_word_count += line_word_count

    if current_chunk_lines:
        chunks.append("\n".join(current_chunk_lines))

    return chunks


def build_vector_store(knowledge_folder=None, vector_folder=None, model_name=DEFAULT_MODEL_NAME):
    """
    Scans knowledge base directory, creates L2-normalized numpy embeddings and saves metadata.
    """
    knowledge_folder = knowledge_folder or DEFAULT_KNOWLEDGE_FOLDER
    vector_folder = vector_folder or DEFAULT_VECTOR_FOLDER

    os.makedirs(knowledge_folder, exist_ok=True)
    os.makedirs(vector_folder, exist_ok=True)

    documents = []
    metadata = []
    files_processed = 0

    print(f"Scanning knowledge folder: {knowledge_folder}")

    for file_name in sorted(os.listdir(knowledge_folder)):
        file_path = os.path.join(knowledge_folder, file_name)
        if not os.path.isfile(file_path):
            continue

        text = ""
        _, ext = os.path.splitext(file_name)
        ext = ext.lower()

        if ext == ".txt":
            encodings = ["utf-8", "utf-8-sig", "latin-1"]
            for enc in encodings:
                try:
                    with open(file_path, "r", encoding=enc) as f:
                        text = f.read()
                    break
                except UnicodeDecodeError:
                    continue
        elif ext == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                pages = [p.extract_text() for p in reader.pages if p.extract_text()]
                text = "\n\n".join(pages)
            except Exception as e:
                print(f"Error reading PDF {file_name}: {e}")
                continue
        else:
            continue

        if not text.strip():
            continue

        print(f"Processing: {file_name} ({len(text)} chars)")
        chunks = chunk_text(text)
        print(f"  Generated {len(chunks)} chunks")

        for idx, chunk in enumerate(chunks):
            documents.append(chunk)
            metadata.append({
                "source": file_name,
                "chunk_id": idx,
                "total_chunks": len(chunks)
            })

        files_processed += 1

    total_chunks = len(documents)
    print(f"\nTotal documents processed: {files_processed}")
    print(f"Total chunks created: {total_chunks}")

    if total_chunks == 0:
        print("Warning: No documents found to index.")
        return {"status": "empty", "files_processed": 0, "total_chunks": 0}

    # Generate embeddings safely with batching
    model = get_embedding_model(model_name)
    print("Generating embeddings...")
    raw_embeddings = model.encode(documents, batch_size=32, show_progress_bar=False)
    embeddings = np.array(raw_embeddings).astype("float32")

    # Normalize vectors for cosine similarity (norm = 1.0)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normalized_embeddings = embeddings / norms

    # Save normalized embeddings matrix
    np.save(os.path.join(vector_folder, "embeddings.npy"), normalized_embeddings)

    with open(os.path.join(vector_folder, "documents.pkl"), "wb") as f:
        pickle.dump(documents, f)
    with open(os.path.join(vector_folder, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    # Optional FAISS save if available
    try:
        import faiss
        dimension = normalized_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(normalized_embeddings)
        faiss.write_index(index, os.path.join(vector_folder, "index.faiss"))
    except Exception as e:
        print(f"[FAISS optional write skipped]: {e}")

    print(f"Vector database saved to {vector_folder}")
    return {
        "status": "success",
        "files_processed": files_processed,
        "total_chunks": total_chunks
    }


if __name__ == "__main__":
    result = build_vector_store()
    print("Build result:", result)