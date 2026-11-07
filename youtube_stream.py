import re
import time
from typing import Optional, Generator
import pytchat
# youtube_stream.py
import yt_dlp
from googleapiclient.discovery import build
import logging

logger = logging.getLogger(__name__)

# 從常見的 YouTube 網址格式中解析出 11 碼 Video ID
def extract_video_id(url: str) -> Optional[str]:
    """Extract the 11-character Video ID from common YouTube URL formats."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"youtube\.com\/live\/([0-9A-Za-z_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

#  YouTube 直播聊天監聽器
class YouTubeChatListener:
    def __init__(self, video_url: str):
        self.video_id = extract_video_id(video_url)
        if not self.video_id:
            raise ValueError("Invalid YouTube live stream URL. Please ensure it contains a 11-character video ID.")
        self.chat = pytchat.create(video_id=self.video_id, interruptable=False)
        self.last_message_time = time.time()

    def get_latest_chat_batch(self):
        """
        Fetch the latest chat messages from the current buffer and return the most recent up to 10 messages.
        """
        if not self.chat or not self.chat.is_alive():
            return {
                "has_new_message": False,
                "messages": [],           # Recent 10 messages list
                "latest_message": "",
                "author_name": "",
                "idle_seconds": 0
            }

        chat_data = self.chat.get()
        items = chat_data.items

        if items:
            # 取得最近最多 10 則訊息物件
            recent_items = items[-10:]
            
            message_list = [
                {
                    "author": c.author.name,
                    "message": c.message,
                    "timestamp": c.datetime,
                    "avatar": c.author.imageUrl
                }
                for c in recent_items
            ]

            latest_item = recent_items[-1]
            return {
                "has_new_message": True,
                "messages": message_list,              # 包含近 10 則彈幕
                "latest_message": latest_item.message,  # 最新的一則
                "author_name": latest_item.author.name,
                "idle_seconds": 0
            }

        return {
            "has_new_message": False,
            "messages": [],
            "latest_message": "",
            "author_name": "",
            "idle_seconds": 0
        }

# 透過 YouTube Data API v3 取得直播間真實即時在線人數 (concurrentViewers)
def fetch_youtube_concurrent_viewers(api_key: str, video_id: str) -> int:
        """
        Fetch the real-time concurrent viewers count for a YouTube live stream using the YouTube Data API v3.
        Quota Consumption: Each call consumes 1 Unit.
        """
        if not api_key or not video_id:
            return 0

        try:
            youtube = build("youtube", "v3", developerKey=api_key)
            request = youtube.videos().list(
                part="liveStreamingDetails",
                id=video_id
            )
            response = request.execute()
            
            items = response.get("items", [])
            if items:
                live_details = items[0].get("liveStreamingDetails", {})
                # 官方真實人數欄位
                concurrent_viewers = live_details.get("concurrentViewers")
                if concurrent_viewers is not None:
                    return int(concurrent_viewers)
                
                # 若直播剛結束或轉為重播，可能沒有 concurrentViewers 欄位
                logger.warning("Call failed to find liveStreamingDetails.concurrentViewers, the live stream may not be live or has been converted to a replay.")
                return 0
            return 0
        except Exception as e:
            logger.error(f"Failed to fetch YouTube real-time viewer count: {str(e)}")
            return 0