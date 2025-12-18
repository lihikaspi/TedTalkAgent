import pandas as pd
import os
import time
from config import (OPENAI_API_KEY,PC_API_KEY, MODEL_BASE_URL, CHAT_MODEL, EMBED_MODEL,
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
    ted_talks = pd.read_csv("ted_talks_en.csv")
    # ted_talks = ted_talks.head(5)

    ted_talks["summary"] = ted_talks.apply(get_talk_summary, axis=1)
    ted_talks.to_csv("ted_talks_en_with_summary.csv", index=False)
    print("Summaries saved to ted_talks_en_with_summary.csv")


def prepare_embeds():
    ted_talks = pd.read_csv("ted_talks_en_with_summary.csv")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,  # Useful for referencing where in the transcript the chunk is
        separators=["\n\n", "\n", ".", " ", ""]
    )

    all_docs = []

    for _, row in ted_talks.iterrows():
        transcript = str(row.get("transcript", ""))
        summary = str(row.get("summary", ""))

        base_metadata = {
            "talk_id": row["talk_id"],
            "title": row["title"],
            "speaker": row["speaker_1"],
            "url": row["url"],
            "topics": row["topics"]
        }

        summary_doc = Document(
            page_content=f"TED Talk Summary: {row['title']}\n\n{summary}",
            metadata={**base_metadata, "content_type": "summary"}
        )
        all_docs.append(summary_doc)

        chunks = text_splitter.split_text(transcript)

        for i, chunk_text in enumerate(chunks):
            contextualized_text = f"Talk: {row['title']}\nSpeaker: {row['speaker_1']}\nContext: {summary[:150]}...\n\nTranscript Snippet: {chunk_text}"

            chunk_doc = Document(
                page_content=contextualized_text,
                metadata={
                    **base_metadata,
                    "content_type": "transcript_chunk",
                    "chunk_id": i
                }
            )
            all_docs.append(chunk_doc)

    print(f"Total documents created: {len(all_docs)}")

    return all_docs


def sync_to_pinecone(documents, embeddings):
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

    return vectorstore


def main():
    if not os.path.exists("ted_talks_en_with_summary.csv"):
        add_summary_col()

    documents = prepare_embeds()

    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        base_url=MODEL_BASE_URL,
        model=EMBED_MODEL
    )

    if documents:
        sync_to_pinecone(documents, embeddings)


if __name__ == "__main__":
    main()