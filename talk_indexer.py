import pandas as pd
import os
import time
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import (OPENAI_API_KEY, PC_API_KEY, MODEL_BASE_URL, EMBED_MODEL,
                    CHUNK_SIZE, CHUNK_OVERLAP, INDEX_NAME, EMBED_SIZE)

# Config for stable indexing
UPLOAD_BATCH_SIZE = 100


def main():
    target_file = "ted_talks_en_with_summary.csv"
    indexed_log = "indexed_talks.txt"

    if not os.path.exists(target_file):
        print("Error: Summarized CSV not found. Please run generate_summaries.py first.")
        return

    # Load summaries and tracking log
    ted_talks = pd.read_csv(target_file)
    existing_ids = set()
    if os.path.exists(indexed_log):
        with open(indexed_log, "r") as f:
            existing_ids = {line.strip() for line in f}

    # Only index talks not already in Pinecone
    new_to_index = ted_talks[~ted_talks['talk_id'].astype(str).isin(existing_ids)]

    if new_to_index.empty:
        print("All summarized talks are already indexed in Pinecone.")
        return

    print(f"Preparing and indexing {len(new_to_index)} talks (Full Transcripts)...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True
    )

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, base_url=MODEL_BASE_URL, model=EMBED_MODEL)
    pc = Pinecone(api_key=PC_API_KEY)

    # Index setup
    if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(name=INDEX_NAME, dimension=EMBED_SIZE, metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(2)

    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings, pinecone_api_key=PC_API_KEY)

    # Process each talk: Chunk full transcript -> Batch Upload
    for _, row in tqdm(new_to_index.iterrows(), total=len(new_to_index), desc="Indexing"):
        t_id = str(row["talk_id"])
        talk_docs = []
        base_meta = {"talk_id": t_id, "title": str(row["title"]), "speaker": str(row["speaker_1"])}

        # 1. Document for Summary
        talk_docs.append(Document(
            page_content=f"SUMMARY: {row['summary']}",
            metadata={**base_meta, "content_type": "summary"}
        ))

        # 2. Documents for FULL Transcript
        full_transcript = str(row.get("transcript", ""))
        chunks = text_splitter.split_text(full_transcript)
        for i, chunk in enumerate(chunks):
            talk_docs.append(Document(
                page_content=f"TRANSCRIPT CHUNK: {chunk}",
                metadata={**base_meta, "content_type": "transcript", "chunk_id": i}
            ))

        # Batch upload to Pinecone
        for j in range(0, len(talk_docs), UPLOAD_BATCH_SIZE):
            vectorstore.add_documents(talk_docs[j: j + UPLOAD_BATCH_SIZE])

        # Log success so we don't repeat this talk if the script restarts
        with open(indexed_log, "a") as f:
            f.write(f"{t_id}\n")

    print("Indexing finished! Your vector store is fully populated.")


if __name__ == "__main__":
    main()