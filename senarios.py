# scenarios.py
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ScenarioConfig:
    name: str
    role_description: str
    sample_stt: List[str]
    mock_chat_pool: List[str]
    default_quick_actions: List[str]

SCENARIO_PRESETS: Dict[str, ScenarioConfig] = {
    "E-Commerce (帶貨直播)": ScenarioConfig(
        name="E-Commerce",
        role_description="You are a high-conversion e-commerce co-host. Focus on product specs, limited-time discounts, coupon redemption, and stock urgency.",
        sample_stt=[
            "This mechanical keyboard features hot-swappable switches and PBT keycaps.",
            "Click the yellow link below for an extra $20 coupon valid for the next 10 minutes only!"
        ],
        mock_chat_pool=["Does it support Mac layout?", "What is the return policy?", "Is shipping to the US free?"],
        default_quick_actions=["🛒 Claim Coupon", "📦 Shipping Policy", "⚡ Product Specs"]
    ),
    "Gaming / Esports (電競/遊戲)": ScenarioConfig(
        name="Gaming / Esports",
        role_description="You are an energetic esports caster sidekick. Hype up game highlights, clutch plays, team stats, and hype chatters.",
        sample_stt=[
            "Unbelievable 1v3 clutch in the final round! He saved the match with 10 HP left!",
            "Let's check the team standings before jumping into Map 3."
        ],
        mock_chat_pool=["WHAT A PLAY!!!", "GG WP", "What sensitivity settings is he using?", "LUL that miss"],
        default_quick_actions=["🔥 Hype Clip", "📊 Player Loadout", "📅 Tournament Schedule"]
    ),
    "Tech Keynote / Launch (產品發表會)": ScenarioConfig(
        name="Tech Launch",
        role_description="You are a polished tech event moderator. Emphasize architecture details, developer benchmarks, ecosystem availability, and NDA rules.",
        sample_stt=[
            "Our new NPU delivers up to 45 TOPS of on-device AI inference with 40% lower power consumption.",
            "SDK release candidate is now live on our developer portal."
        ],
        mock_chat_pool=["Is it open source?", "When is the developer beta available?", "How does it compare to M3 chips?"],
        default_quick_actions=["📄 Read Whitepaper", "💻 API Documentation", "🗓️ Release Timeline"]
    ),
    "IRL & Casual Chat (休閒雜談)": ScenarioConfig(
        name="IRL & Casual",
        role_description="You are an empathetic, witty conversational buddy. Match casual vibes, respond to daily life stories, and handle trolls gently.",
        sample_stt=[
            "I finally tried that ramen place in Shibuya today, broth was insanely rich.",
            "What should we plan for this weekend's subathon stream?"
        ],
        mock_chat_pool=["Any travel plans for next month?", "That ramen looked delicious!", "How long have you been streaming today?"],
        default_quick_actions=["🍜 Streamer FAQ", "🎵 Song Request", "💬 Today's Topic"]
    )
}