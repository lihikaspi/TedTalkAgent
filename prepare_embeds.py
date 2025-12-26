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
    """
    Summarizes the talk using the LLM.
    Handles Azure OpenAI Content Safety/Security errors by using the original description as a fallback.
    """
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=MODEL_BASE_URL,
        model=CHAT_MODEL,
        max_retries=2
    )

    prompt = f"""
    Title: {row['title']}
    Topics: {row['topics']}
    Transcript Snippet: {str(row['transcript'])[:8000]}

    Summarize this TED talk in one paragraph (under 200 words). 
    Focus on the main thesis and key takeaway.
    """

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        error_str = str(e).lower()
        # Fallback for Security/Safety filters (common in political/social TED topics)
        if any(msg in error_str for msg in ["content_filter", "safety", "400", "policy", "security"]):
            print(f"Safety Filter triggered for '{row['title']}'. Using original description fallback.")
            # Use original description from dataset as the summary
            return row.get('description', f"A TED talk titled '{row['title']}' covering {row['topics']}.")

        print(f"Error for {row['title']}: {e}")
        return ""


def add_summary_col():
    """Adds a summary column incrementally to the dataset."""
    target_file = "ted_talks_en_with_summary.csv"
    if not os.path.exists("ted_talks_en.csv"):
        print("Error: Source ted_talks_en.csv not found.")
        return

    source_df = pd.read_csv("ted_talks_en.csv")

    if os.path.exists(target_file):
        existing_df = pd.read_csv(target_file)
        new_talks = source_df[~source_df['talk_id'].isin(existing_df['talk_id'])].copy()
        if new_talks.empty:
            print("All summaries are up to date.")
            return
        print(f"Processing {len(new_talks)} new summaries...")
        new_talks["summary"] = new_talks.apply(get_talk_summary, axis=1)
        pd.concat([existing_df, new_talks]).to_csv(target_file, index=False)
    else:
        print("Generating initial summaries...")
        source_df["summary"] = source_df.apply(get_talk_summary, axis=1)
        source_df.to_csv(target_file, index=False)


def prepare_embeds(exclude_ids=None):
    """
    Chunks data into 'summary' and 'transcript' types.
    Summaries help with Goal 1 (Finding a talk) and Goal 2 (Listing 3 talks).
    Chunks help with Goal 3 (Summarizing ideas) and Goal 4 (Evidence).
    """
    exclude_ids = exclude_ids or set()
    ted_talks = pd.read_csv("ted_talks_en_with_summary.csv")

    # Filter out already processed IDs
    ted_talks = ted_talks[~ted_talks['talk_id'].astype(str).isin(exclude_ids)]
    print(f"Processing {len(ted_talks)} talks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True
    )

    all_docs = []
    processed_ids = []

    for _, row in ted_talks.iterrows():
        t_id = str(row["talk_id"])
        base_meta = {
            "talk_id": t_id,
            "title": row["title"],
            "speaker": row["speaker_1"],
            "url": row["url"],
            "topics": row["topics"]
        }

        # 1. Store the High-Level Summary (Supports Goal 1 and Goal 2)
        summary_text = str(row.get("summary", ""))
        all_docs.append(Document(
            page_content=f"TED TALK OVERVIEW\nTitle: {row['title']}\nSpeaker: {row['speaker_1']}\nSummary: {summary_text}",
            metadata={**base_meta, "content_type": "summary"}
        ))

        # 2. Store Transcript Chunks (Supports Goal 3 and Goal 4)
        transcript = str(row.get("transcript", ""))
        chunks = text_splitter.split_text(transcript)
        for i, chunk in enumerate(chunks):
            # We inject the title and speaker into every chunk to ensure retrieval context is clear
            contextualized_chunk = f"TALK: {row['title']} by {row['speaker_1']}\nTRANSCRIPT SNIPPET: {chunk}"
            all_docs.append(Document(
                page_content=contextualized_chunk,
                metadata={**base_meta, "content_type": "transcript", "chunk_id": i}
            ))

        processed_ids.append(t_id)

    return all_docs, processed_ids


def sync_to_pinecone(documents, embeddings, processed_ids):
    if not documents: return
    pc = Pinecone(api_key=PC_API_KEY)

    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME, dimension=EMBED_SIZE, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(INDEX_NAME).status['ready']: time.sleep(1)

    PineconeVectorStore.from_documents(
        documents=documents, index_name=INDEX_NAME,
        embedding=embeddings, pinecone_api_key=PC_API_KEY
    )

    with open("indexed_talks.txt", "a") as f:
        for t_id in processed_ids: f.write(f"{t_id}\n")


def main():
    add_summary_col()

    indexed_file = "indexed_talks.txt"
    existing_ids = set()
    if os.path.exists(indexed_file):
        with open(indexed_file, "r") as f:
            existing_ids = {line.strip() for line in f}

    docs, new_ids = prepare_embeds(exclude_ids=existing_ids)

    if docs:
        embeddings = OpenAIEmbeddings(
            api_key=OPENAI_API_KEY, base_url=MODEL_BASE_URL, model=EMBED_MODEL
        )
        sync_to_pinecone(docs, embeddings, new_ids)
        print("Success: New talks indexed.")
    else:
        print("No new content to index.")


if __name__ == "__main__":
    main()