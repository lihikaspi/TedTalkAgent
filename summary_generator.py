import pandas as pd
import os
import concurrent.futures
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from config import (OPENAI_API_KEY, MODEL_BASE_URL, CHAT_MODEL)

# Configuration for cost/time efficiency
MAX_TRANSCRIPT_CHARS = 6000
MAX_WORKERS = 5
CHECKPOINT_INTERVAL = 20


def get_talk_summary(row, llm):
    """Summarizes a single talk with security/safety fallback."""
    clean_title = str(row['title']).replace('"', "'")
    transcript_snippet = str(row['transcript'])[:MAX_TRANSCRIPT_CHARS]

    prompt = f"""
    Title: {clean_title}
    Topics: {row['topics']}
    Transcript Snippet: {transcript_snippet}

    Summarize this TED talk in one paragraph (under 150 words). 
    Focus on the main thesis and key takeaway.
    """
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        error_str = str(e).lower()
        if any(msg in error_str for msg in ["content_filter", "safety", "policy", "security", "400"]):
            return f"[Safety Fallback]: {row.get('description', 'A TED talk overview.')}"
        return ""


def main():
    target_file = "ted_talks_en_with_summary.csv"
    source_file = "ted_talks_en.csv"

    if not os.path.exists(source_file):
        print(f"Error: {source_file} not found.")
        return

    source_df = pd.read_csv(source_file)

    # Load existing progress or initialize with the source schema
    if os.path.exists(target_file):
        df_progress = pd.read_csv(target_file)
        existing_ids = set(df_progress['talk_id'].astype(str))
    else:
        # Create empty df with same columns as source plus summary
        df_progress = pd.DataFrame(columns=source_df.columns.tolist() + ['summary'])
        existing_ids = set()

    # Filter out already summarized talks
    new_talks = source_df[~source_df['talk_id'].astype(str).isin(existing_ids)].copy()

    if new_talks.empty:
        print("All talks are already summarized in the CSV.")
        return

    llm = ChatOpenAI(api_key=OPENAI_API_KEY, base_url=MODEL_BASE_URL, model=CHAT_MODEL)
    print(f"Summarizing {len(new_talks)} talks using {MAX_WORKERS} threads...")

    results_map = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_id = {executor.submit(get_talk_summary, row, llm): row['talk_id'] for _, row in new_talks.iterrows()}
        pbar = tqdm(total=len(new_talks), desc="Summarizing")

        for i, future in enumerate(concurrent.futures.as_completed(future_to_id)):
            t_id = future_to_id[future]
            try:
                results_map[t_id] = future.result()
            except:
                results_map[t_id] = ""

            pbar.update(1)

            # Checkpoint save
            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                # Map results to the local new_talks copy
                for tid, summ in results_map.items():
                    new_talks.loc[new_talks['talk_id'] == tid, 'summary'] = summ

                # Filter for talks that now have summaries
                processed_this_session = new_talks[new_talks['summary'].notna()].copy()

                # Robust concatenation to avoid FutureWarnings
                if not processed_this_session.empty:
                    if df_progress.empty:
                        combined_df = processed_this_session
                    else:
                        combined_df = pd.concat([df_progress, processed_this_session], ignore_index=True)

                    combined_df.drop_duplicates('talk_id', inplace=True)
                    combined_df.to_csv(target_file, index=False)

    # Final mapping and save
    for tid, summ in results_map.items():
        new_talks.loc[new_talks['talk_id'] == tid, 'summary'] = summ

    if df_progress.empty:
        final_df = new_talks
    else:
        final_df = pd.concat([df_progress, new_talks], ignore_index=True)

    final_df.drop_duplicates('talk_id', inplace=True)
    final_df.to_csv(target_file, index=False)
    print(f"Success! Summaries saved to {target_file}")


if __name__ == "__main__":
    main()