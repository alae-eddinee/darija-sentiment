"""
YouTube comment scraper for Moroccan Darija sentiment analysis.

Supports two modes:
  1. Direct web scraping via youtube-comment-downloader (DEFAULT, NO API KEY REQUIRED)
  2. Official YouTube Data API v3 (if YOUTUBE_API_KEY is provided and --use-api is passed)

Output CSV: comment_id, video_id, text, published_at, like_count
"""

import os
import csv
import time
import argparse
from pathlib import Path
from itertools import islice
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OUTPUT_FIELDS = ["comment_id", "video_id", "text", "published_at", "like_count"]


def load_existing_comment_ids(output_path: Path) -> tuple[set, dict]:
    """Load already-downloaded comment IDs and counts per video_id to avoid duplicates."""
    ids = set()
    counts = {}
    if output_path.exists() and output_path.stat().st_size > 0:
        try:
            with open(output_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = row.get("comment_id")
                    vid = row.get("video_id")
                    if cid:
                        ids.add(cid)
                    if vid:
                        counts[vid] = counts.get(vid, 0) + 1
        except Exception as e:
            print(f"  Notice while loading existing IDs: {e}")
    return ids, counts


def save_comments(comments: list[dict], output_path: Path):
    """Save comments to CSV, appending if file exists."""
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    mode = "a" if file_exists else "w"
    
    with open(output_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        if mode == "w" or not file_exists:
            writer.writeheader()
        writer.writerows(comments)


def scrape_without_api(video_ids: list[str], output_path: Path, max_per_video: int = 250, target_total: int = 3000) -> int:
    """
    Scrape comments without any YouTube API key using youtube-comment-downloader.
    """
    try:
        from youtube_comment_downloader import YoutubeCommentDownloader
    except ImportError:
        print("❌ youtube-comment-downloader is not installed. Run: pip install youtube-comment-downloader")
        return 0

    downloader = YoutubeCommentDownloader()
    existing_ids, video_counts = load_existing_comment_ids(output_path)
    total_new = 0

    print(f"📥 Starting non-API scrape across {len(video_ids)} videos...")
    print(f"   Already have {len(existing_ids)} comments on disk across {len(video_counts)} videos. Target: {target_total} comments.")

    for i, vid in enumerate(video_ids, 1):
        vid = vid.strip()
        if not vid or vid.startswith("#"):
            continue

        if len(existing_ids) >= target_total:
            print(f"🎯 Target of {target_total} total comments reached!")
            break

        if video_counts.get(vid, 0) >= max_per_video:
            print(f"⏭️  [{i}/{len(video_ids)}] Video {vid} already has {video_counts[vid]} comments, skipping.")
            continue

        print(f"\n[{i}/{len(video_ids)}] Scraping video: https://www.youtube.com/watch?v={vid}")
        batch = []
        try:
            comments = downloader.get_comments_from_url(f"https://www.youtube.com/watch?v={vid}", sort_by=0)
            for comment in comments:
                cid = comment.get("cid")
                if not cid or cid in existing_ids:
                    continue

                text = comment.get("text", "").strip()
                if not text:
                    continue

                batch.append({
                    "comment_id": cid,
                    "video_id": vid,
                    "text": text,
                    "published_at": comment.get("time", ""),
                    "like_count": comment.get("votes", 0)
                })
                existing_ids.add(cid)

                if len(batch) >= max_per_video:
                    break

            if batch:
                save_comments(batch, output_path)
                total_new += len(batch)
                print(f"  ✅ Saved {len(batch)} new comments (Total new: {total_new}, Total on disk: {len(existing_ids)})")
            else:
                print("  ⏭️  No new comments found or comments disabled.")

        except Exception as e:
            print(f"  ⚠️  Error scraping video {vid}: {e}")
            time.sleep(1)

    print(f"\n🏁 Scrape finished! Added {total_new} new comments.")
    print(f"   File: {output_path} (Total records: {len(existing_ids)})")
    return total_new


def scrape_with_api(video_ids: list[str], output_path: Path, api_key: str, target_total: int = 3000) -> int:
    """Scrape using official YouTube Data API v3 (fallback if user provides key)."""
    try:
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
    except ImportError:
        print("google-api-python-client not installed.")
        return 0

    youtube = build("youtube", "v3", developerKey=api_key)
    existing_ids = load_existing_comment_ids(output_path)
    total_new = 0

    print(f"📥 Starting official YouTube API scrape across {len(video_ids)} videos...")

    for i, vid in enumerate(video_ids, 1):
        vid = vid.strip()
        if not vid or vid.startswith("#"):
            continue

        if len(existing_ids) + total_new >= target_total:
            break

        print(f"\n[{i}/{len(video_ids)}] Fetching comments via API for: {vid}")
        next_page = None
        vid_comments = []

        while True:
            try:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=vid,
                    maxResults=100,
                    pageToken=next_page,
                    textFormat="plainText"
                )
                response = request.execute()
                for item in response.get("items", []):
                    snippet = item["snippet"]["topLevelComment"]["snippet"]
                    cid = item["snippet"]["topLevelComment"]["id"]
                    if cid not in existing_ids:
                        vid_comments.append({
                            "comment_id": cid,
                            "video_id": vid,
                            "text": snippet["textDisplay"],
                            "published_at": snippet["publishedAt"],
                            "like_count": snippet.get("likeCount", 0)
                        })
                        existing_ids.add(cid)

                next_page = response.get("nextPageToken")
                if not next_page or len(vid_comments) >= 200:
                    break
            except Exception as e:
                print(f"  API error for {vid}: {e}")
                break

        if vid_comments:
            save_comments(vid_comments, output_path)
            total_new += len(vid_comments)
            print(f"  ✅ Saved {len(vid_comments)} comments (Total new: {total_new})")

    return total_new


def main():
    parser = argparse.ArgumentParser(description="Scrape YouTube comments for Moroccan Darija sentiment analysis")
    parser.add_argument("--video-ids", "-v", type=str, default="data/video_ids.txt",
                        help="Path to file with video IDs")
    parser.add_argument("--output", "-o", type=str, default="data/raw/youtube_comments/comments.csv",
                        help="Output CSV path")
    parser.add_argument("--target", "-t", type=int, default=3000,
                        help="Target number of comments")
    parser.add_argument("--use-api", action="store_true",
                        help="Force using official YouTube Data API instead of direct scraper")
    args = parser.parse_args()

    video_ids_path = Path(args.video_ids)
    if not video_ids_path.exists():
        print(f"❌ Video IDs file not found: {video_ids_path}")
        return

    with open(video_ids_path, "r", encoding="utf-8") as f:
        video_ids = [line.split("#")[0].strip() for line in f if line.strip() and not line.strip().startswith("#")]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("YOUTUBE_API_KEY")
    if args.use_api and api_key:
        scrape_with_api(video_ids, output_path, api_key, target_total=args.target)
    else:
        scrape_without_api(video_ids, output_path, max_per_video=250, target_total=args.target)


if __name__ == "__main__":
    main()
