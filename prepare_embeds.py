import pandas as pd
import os
import time
from config import (OPENAI_API_KEY, PC_API_KEY, MODEL_BASE_URL, CHAT_MODEL, EMBED_MODEL,
                    CHUNK_SIZE, CHUNK_OVERLAP, INDEX_NAME, EMBED_SIZE)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


def get_talk_summary(row):
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=MODEL_BASE_URL,
        model=CHAT_MODEL
    )

    prompt = f"""
    This is a transcript of a TED talk titled "{row['title']}" regarding the topics {row['topics']}:

    {row['transcript']}

    Write one paragraph summarizing the key points of the TED talk using under 200 words.
    """

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"Error summarizing {row['title']}: {e}")
        return ""


def add_summary_col():
    """Only summarizes talks that haven't been summarized yet."""
    target_file = "ted_talks_en_with_summary.csv"
    source_df = pd.read_csv("ted_talks_en.csv")

    if os.path.exists(target_file):
        existing_df = pd.read_csv(target_file)
        # Find IDs that are in source but NOT in existing
        new_talks = source_df[~source_df['talk_id'].isin(existing_df['talk_id'])]

        if new_talks.empty:
            print("All talks in source already have summaries.")
            return

        print(f"Found {len(new_talks)} new talks to summarize.")
        new_talks["summary"] = new_talks.apply(get_talk_summary, axis=1)

        # Combine and save
        updated_df = pd.concat([existing_df, new_talks], ignore_index=True)
        updated_df.to_csv(target_file, index=False)
    else:
        print("Creating new summary file...")
        source_df["summary"] = source_df.apply(get_talk_summary, axis=1)
        source_df.to_csv(target_file, index=False)

    print(f"Summaries updated in {target_file}")


def get_existing_ids_in_pinecone():
    """
    Fetches a sample of existing IDs to check if data is already there.
    Note: For very large datasets, checking 'existence' is better handled
    by local tracking or the LangChain Indexing API.
    """
    pc = Pinecone(api_key=PC_API_KEY)
    if INDEX_NAME not in pc.list_indexes().names():
        return set()

    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    # We use metadata filtering to find which talk_ids are present
    # However, Pinecone doesn't allow 'list all unique metadata values' easily.
    # A robust way is keeping a local file 'indexed_talks.txt'
    if os.path.exists("indexed_talks.txt"):
        with open("indexed_talks.txt", "r") as f:
            return set(line.strip() for line in f)
    return set()


def prepare_embeds(exclude_ids=None):
    """Chunks transcripts, excluding talk_ids that are already indexed."""
    exclude_ids = exclude_ids or set()
    ted_talks = pd.read_csv("ted_talks_en_with_summary.csv")

    # Filter out already processed IDs
    initial_count = len(ted_talks)
    ted_talks = ted_talks[~ted_talks['talk_id'].astype(str).isin(exclude_ids)]
    print(f"Processing {len(ted_talks)} talks (Skipped {initial_count - len(ted_talks)} already indexed).")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    all_docs = []
    processed_ids_this_run = []

    for _, row in ted_talks.iterrows():
        t_id = str(row["talk_id"])
        transcript = str(row.get("transcript", ""))
        summary = str(row.get("summary", ""))

        base_metadata = {
            "talk_id": t_id,
            "title": row["title"],
            "speaker": row["speaker_1"],
            "url": row["url"],
            "topics": row["topics"]
        }

        # Summary doc
        all_docs.append(Document(
            page_content=f"TED Talk Summary: {row['title']}\n\n{summary}",
            metadata={**base_metadata, "content_type": "summary"}
        ))

        # Transcript chunks
        chunks = text_splitter.split_text(transcript)
        for i, chunk_text in enumerate(chunks):
            contextualized_text = f"Talk: {row['title']}\nSpeaker: {row['speaker_1']}\nContext: {summary[:150]}...\n\nTranscript Snippet: {chunk_text}"
            all_docs.append(Document(
                page_content=contextualized_text,
                metadata={**base_metadata, "content_type": "transcript_chunk", "chunk_id": i}
            ))

        processed_ids_this_run.append(t_id)

    return all_docs, processed_ids_this_run


def sync_to_pinecone(documents, embeddings, processed_ids):
    if not documents:
        print("No new documents to upload.")
        return None

    pc = Pinecone(api_key=PC_API_KEY)

    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_SIZE,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)
            print("Waiting for index to initialize...")

    print(f"Uploading {len(documents)} documents to Pinecone index '{INDEX_NAME}'...")

    vectorstore = PineconeVectorStore.from_documents(
        documents=documents,
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=PC_API_KEY
    )

    # Update local tracking file
    with open("indexed_talks.txt", "a") as f:
        for t_id in processed_ids:
            f.write(f"{t_id}\n")

    return vectorstore


def main():
    # 1. Update summaries (incremental)
    add_summary_col()

    # 2. Check what's already in Pinecone (via local manifest)
    existing_ids = get_existing_ids_in_pinecone()

    # 3. Prepare only new docs
    documents, new_ids = prepare_embeds(exclude_ids=existing_ids)

    # 4. Embed and Sync
    if documents:
        embeddings = OpenAIEmbeddings(
            api_key=OPENAI_API_KEY,
            base_url=MODEL_BASE_URL,
            model=EMBED_MODEL
        )
        sync_to_pinecone(documents, embeddings, new_ids)
        print("Sync complete.")
    else:
        print("Everything is already up to date.")


if __name__ == "__main__":
    main()