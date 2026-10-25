"""
YouTube comment scraper for Moroccan Darija sentiment analysis.

Uses YouTube Data API v3 to collect comments from specified videos.
Handles quota limits, pagination, and deduplication.
"""

import os
import csv
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:
    print("google-api-python-client not installed. Run: pip install google-api-python-client")
    raise


# Load environment variables
load_dotenv()

# Constants
MAX_RESULTS_PER_PAGE = 100
QUOTA_WAIT_SECONDS = 60
OUTPUT_FIELDS = ["comment_id", "video_id", "text", "published_at", "like_count"]


def get_youtube_client(api_key: str):
    """Build a YouTube Data API v3 client."""
    return build("youtube", "v3", developerKey=api_key)


def load_existing_comment_ids(output_path: Path) -> set:
    """Load already-downloaded comment IDs to avoid duplicates."""
    ids = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ids.add(row["comment_id"])
    return ids


def fetch_comments(youtube, video_id: str, existing_ids: set) -> list[dict]:
    """
    Fetch all top-level comments for a video.
    
    Handles pagination and skips already-downloaded comments.
    Returns list of comment dicts.
    """
    comments = []
    next_page_token = None
    
    while True:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=MAX_RESULTS_PER_PAGE,
                pageToken=next_page_token,
                textFormat="plainText",
                order="relevance"
            )
            response = request.execute()
        except HttpError as e:
            if e.resp.status == 403:
                print(f"  ⚠️  Quota exceeded or comments disabled for {video_id}. "
                      f"Waiting {QUOTA_WAIT_SECONDS}s...")
                if "commentsDisabled" in str(e):
                    print(f"  ❌ Comments disabled for {video_id}, skipping.")
                    return comments
                time.sleep(QUOTA_WAIT_SECONDS)
                continue
            elif e.resp.status == 404:
                print(f"  ❌ Video {video_id} not found, skipping.")
                return comments
            else:
                print(f"  ❌ HTTP error {e.resp.status} for {video_id}: {e}")
                return comments
        
        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comment_id = item["snippet"]["topLevelComment"]["id"]
            
            if comment_id in existing_ids:
                continue
            
            comments.append({
                "comment_id": comment_id,
                "video_id": video_id,
                "text": snippet["textDisplay"],
                "published_at": snippet["publishedAt"],
                "like_count": snippet.get("likeCount", 0),
            })
            existing_ids.add(comment_id)
        
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    
    return comments


def save_comments(comments: list[dict], output_path: Path, append: bool = True):
    """Save comments to CSV, appending if file exists."""
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    mode = "a" if (append and file_exists) else "w"
    
    with open(output_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        if mode == "w" or not file_exists:
            writer.writeheader()
        writer.writerows(comments)


def scrape_videos(video_ids: list[str], output_path: Path, api_key: str) -> int:
    """
    Scrape comments from a list of YouTube video IDs.
    
    Args:
        video_ids: List of YouTube video IDs
        output_path: Path to output CSV file
        api_key: YouTube Data API key
    
    Returns:
        Total number of new comments scraped
    """
    youtube = get_youtube_client(api_key)
    existing_ids = load_existing_comment_ids(output_path)
    total_new = 0
    
    print(f"📥 Starting scrape of {len(video_ids)} videos...")
    print(f"   Already have {len(existing_ids)} comments on disk.")
    
    for i, vid in enumerate(video_ids, 1):
        vid = vid.strip()
        if not vid or vid.startswith("#"):
            continue
        
        print(f"\n[{i}/{len(video_ids)}] Fetching comments for: {vid}")
        comments = fetch_comments(youtube, vid, existing_ids)
        
        if comments:
            save_comments(comments, output_path)
            total_new += len(comments)
            print(f"  ✅ Got {len(comments)} new comments (total new: {total_new})")
        else:
            print(f"  ⏭️  No new comments.")
    
    print(f"\n🏁 Done! Scraped {total_new} new comments total.")
    print(f"   Output: {output_path}")
    return total_new


def main():
    """CLI entry point for the YouTube comment scraper."""
    parser = argparse.ArgumentParser(description="Scrape YouTube comments for Darija sentiment analysis")
    parser.add_argument(
        "--video-ids", "-v",
        type=str,
        default="data/video_ids.txt",
        help="Path to file with video IDs (one per line)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/raw/youtube_comments/comments.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--api-key", "-k",
        type=str,
        default=None,
        help="YouTube API key (or set YOUTUBE_API_KEY env var)"
    )
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        print("❌ No API key found. Set YOUTUBE_API_KEY in .env or pass --api-key")
        return
    
    video_ids_path = Path(args.video_ids)
    if not video_ids_path.exists():
        print(f"❌ Video IDs file not found: {video_ids_path}")
        return
    
    with open(video_ids_path, "r") as f:
        video_ids = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    scrape_videos(video_ids, output_path, api_key)


if __name__ == "__main__":
    main()
