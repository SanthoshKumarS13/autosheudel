# config.py

import os

# --- API Keys ---
# WARNING: Embedding API keys directly in code is NOT recommended for production environments.
# For better security, always use environment variables or a more secure secrets management solution.
# Fetch keys from environment variables for deployment.
# Default values are provided for local testing or if environment variables are not set.

# --- NEW: Hugging Face Configuration ---
# The script will use the HF_TOKEN secret from your GitHub repository settings or a local .env file.
HUGGING_FACE_TOKEN = os.getenv("HF_TOKEN", "YOUR_HUGGING_FACE_TOKEN")
INFERENCE_API_ENDPOINTS = [
    "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell",
    "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3-medium-diffusers",
    "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0",
    "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5",
    "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
]

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_DEEPSEEK_API_KEY") # Replace with your actual key if not using env var
OPENROUTER_MODEL = "deepseek/deepseek-chat-v3-0324:free"
OPENROUTER_SITE_URL = "https://example.com" # Your site URL for OpenRouter (optional)
OPENROUTER_SITE_NAME = "Insight Pulse" # Your site name for OpenRouter (optional)

OPENROUTER_MISTRAL_API_KEY = os.getenv("OPENROUTER_MISTRAL_API_KEY", "YOUR_OPENROUTER_MISTRAL_API_KEY") # Replace with your actual key if not using env var
OPENROUTER_MISTRAL_MODEL = "mistralai/mistral-small-3.2-24b-instruct:free"

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "YOUR_PEXELS_API_KEY") # Replace with your actual key if not using env var
PEXELS_API_URL = "https://api.pexels.com/v1"

UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY", "YOUR_UNSPLASH_ACCESS_KEY") # Replace with your actual key if not using env var
UNSPLASH_API_URL = "https://api.unsplash.com"

OPENVERSE_API_URL = "https://api.openverse.engineering/v1/images/" # Openverse (no API key needed for basic use)

PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "YOUR_PIXABAY_API_KEY") # Replace with your actual key if not using env var
PIXABAY_API_URL = "https://pixabay.com/api/"

CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "YOUR_CLOUDINARY_CLOUD_NAME") # Replace with your actual key if not using env var
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY", "YOUR_CLOUDINARY_API_KEY") # Replace with your actual key if not using env var
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "YOUR_CLOUDINARY_API_SECRET") # Replace with your actual key if not using env var

FB_PAGE_ACCESS_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN", "YOUR_FB_PAGE_ACCESS_TOKEN") # Replace with your actual key if not using env var
INSTAGRAM_BUSINESS_ACCOUNT_ID = os.getenv("INSTAGRAM_BUSINESS_ACCOUNT_ID", "YOUR_INSTAGRAM_BUSINESS_ACCOUNT_ID") # Replace with your actual key if not using env var

# --- Output Directories and Files ---
IMAGE_OUTPUT_DIR = "output/images"
JSON_OUTPUT_DIR = "output/json"
EXCEL_OUTPUT_DIR = "output/excel"

ALL_POSTS_JSON_FILE = f"{JSON_OUTPUT_DIR}/all_posts.json"
ALL_POSTS_EXCEL_FILE = f"{EXCEL_OUTPUT_DIR}/all_posts.xlsx"
STYLE_RECOMMENDATIONS_FILE = f"{JSON_OUTPUT_DIR}/style_recommendations.json"

# --- Weekly Analysis Configuration ---
WEEKLY_ANALYSIS_INTERVAL_DAYS = 7

# --- Canvas Dimensions (Instagram Story / Reel size) ---
CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 1350

# --- Font Paths (ensure these paths are correct relative to your script) ---
FONT_PATH_EXTRABOLD = "fonts/Montserrat-ExtraBold.ttf"
FONT_PATH_BOLD = "fonts/Montserrat-Bold.ttf"
FONT_PATH_MEDIUM = "fonts/Montserrat-Medium.ttf"
FONT_PATH_REGULAR = "fonts/Montserrat-Regular.ttf"
FONT_PATH_LIGHT = "fonts/Montserrat-Light.ttf"

# --- Colors (RGBA format) ---
COLOR_GRADIENT_TOP_LEFT = (26, 26, 46, 255)
COLOR_GRADIENT_BOTTOM_RIGHT = (10, 10, 20, 255)

COLOR_WHITE = (255, 255, 255, 255)
COLOR_RED = (255, 70, 70, 255)
COLOR_DARK_GRAY = (50, 50, 50, 255)
COLOR_LIGHT_GRAY_TEXT = (200, 200, 200, 255)

# --- Font Sizes ---
FONT_SIZE_TOP_LEFT_TEXT = 35
FONT_SIZE_TIMESTAMP = 32
FONT_SIZE_HEADLINE = 50
FONT_SIZE_SUMMARY = 40
FONT_SIZE_SOURCE = 30

# --- Padding and Margins ---
LEFT_PADDING = 50
RIGHT_PADDING = 50
TOP_PADDING = 50
BOTTOM_PADDING = 50

TOP_LEFT_TEXT_POS_X = LEFT_PADDING
TOP_LEFT_TEXT_POS_Y = TOP_PADDING

TIMESTAMP_POS_X_RIGHT_ALIGN = CANVAS_WIDTH - RIGHT_PADDING
TIMESTAMP_POS_Y = TOP_PADDING

IMAGE_DISPLAY_WIDTH = CANVAS_WIDTH - (LEFT_PADDING)
IMAGE_DISPLAY_HEIGHT = int(CANVAS_HEIGHT * 0.40)
IMAGE_TOP_MARGIN_FROM_TOP_ELEMENTS = 50
IMAGE_ROUND_RADIUS = 10

TITLE_TOP_MARGIN_FROM_IMAGE = 50
TITLE_MAX_WORDS = 4
TITLE_LINE_SPACING = 7

SUMMARY_TOP_MARGIN_FROM_TITLE = 40
SUMMARY_MIN_WORDS = 40
SUMMARY_MAX_WORDS = 50
SUMMARY_LINE_SPACING = 7
SUMMARY_MAX_LINES = 6
SUMMARY_REGENERATE_ATTEMPTS = 4

# --- Logo ---
LOGO_PATH = "assets/insight_pulse_logo.png"
LOGO_WIDTH = 350
LOGO_HEIGHT = 120

# --- Source Box ---
SOURCE_RECT_PADDING_X = 25
SOURCE_RECT_PADDING_Y = 15
SOURCE_RECT_RADIUS = 15
SOURCE_POS_X_RIGHT_ALIGN = CANVAS_WIDTH - RIGHT_PADDING

# --- Divider Line ---
DIVIDER_Y_OFFSET_FROM_SUMMARY = 50
DIVIDER_LINE_THICKNESS = 5

# --- Workflow State Management ---
CONTENT_TYPE_CYCLE = [
    'world_news',
    'indian_news',
    'tech_news',
    'environmental_news',
    'world_news',
    'indian_news'
]
