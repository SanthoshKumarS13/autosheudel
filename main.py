# main.py

import requests
import json
import os
import pandas as pd
from datetime import datetime, timedelta, UTC
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
import textwrap
import io
import random
import feedparser
import ssl
import re
import sys
import traceback
from openai import OpenAI
import cloudinary
import cloudinary.uploader

# Use a try-except block to gracefully handle missing libraries
try:
    import torch
    from diffusers import FluxPipeline
    import transformers
    FLUX_LIBRARIES_AVAILABLE = True
except ImportError:
    print("Warning: `torch`, `diffusers`, or `transformers` not found. FLUX image generation will be disabled.")
    FLUX_LIBRARIES_AVAILABLE = False


# Fix for some SSL certificate issues with feedparser on some systems
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context


# Import configuration and state manager
from config import (
    OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_SITE_URL, OPENROUTER_SITE_NAME,
    OPENROUTER_MISTRAL_API_KEY, OPENROUTER_MISTRAL_MODEL,
    PEXELS_API_KEY, PEXELS_API_URL,
    UNSPLASH_ACCESS_KEY, UNSPLASH_API_URL,
    OPENVERSE_API_URL,
    PIXABAY_API_KEY, PIXABAY_API_URL,
    CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET,
    FB_PAGE_ACCESS_TOKEN, INSTAGRAM_BUSINESS_ACCOUNT_ID,
    # --- FLUX Config Import ---
    ENABLE_FLUX_IMAGE_GENERATION, FLUX_MODEL_ID, HUGGING_FACE_TOKEN,
    IMAGE_OUTPUT_DIR, JSON_OUTPUT_DIR, EXCEL_OUTPUT_DIR,
    ALL_POSTS_JSON_FILE, ALL_POSTS_EXCEL_FILE, STYLE_RECOMMENDATIONS_FILE,
    WEEKLY_ANALYSIS_INTERVAL_DAYS,
    CANVAS_WIDTH, CANVAS_HEIGHT,
    FONT_PATH_EXTRABOLD, FONT_PATH_BOLD, FONT_PATH_MEDIUM, FONT_PATH_REGULAR, FONT_PATH_LIGHT,
    COLOR_GRADIENT_TOP_LEFT, COLOR_GRADIENT_BOTTOM_RIGHT,
    COLOR_WHITE, COLOR_RED, COLOR_DARK_GRAY, COLOR_LIGHT_GRAY_TEXT,
    FONT_SIZE_TOP_LEFT_TEXT, FONT_SIZE_TIMESTAMP, FONT_SIZE_HEADLINE, FONT_SIZE_SUMMARY, FONT_SIZE_SOURCE,
    LEFT_PADDING, RIGHT_PADDING, TOP_PADDING, BOTTOM_PADDING,
    TOP_LEFT_TEXT_POS_X, TOP_LEFT_TEXT_POS_Y,
    TIMESTAMP_POS_X_RIGHT_ALIGN, TIMESTAMP_POS_Y,
    IMAGE_DISPLAY_WIDTH, IMAGE_DISPLAY_HEIGHT, IMAGE_TOP_MARGIN_FROM_TOP_ELEMENTS, IMAGE_ROUND_RADIUS,
    TITLE_TOP_MARGIN_FROM_IMAGE, TITLE_MAX_WORDS, TITLE_LINE_SPACING,
    SUMMARY_TOP_MARGIN_FROM_TITLE, SUMMARY_MIN_WORDS, SUMMARY_MAX_WORDS, SUMMARY_LINE_SPACING, SUMMARY_MAX_LINES,
    LOGO_PATH, LOGO_WIDTH, LOGO_HEIGHT,
    SOURCE_RECT_PADDING_X, SOURCE_RECT_PADDING_Y, SOURCE_RECT_RADIUS,
    SOURCE_POS_X_RIGHT_ALIGN,
    DIVIDER_Y_OFFSET_FROM_SUMMARY, DIVIDER_LINE_THICKNESS,
    CONTENT_TYPE_CYCLE
)
from state_manager import WorkflowStateManager

# --- Utility Functions ---

def load_font(font_path, size):
    """Loads a font with error handling."""
    try:
        return ImageFont.truetype(font_path, size)
    except IOError:
        print(f"Error: Font file not found at {font_path}. Please ensure the font file exists in the 'fonts' folder.")
        return ImageFont.load_default()
    except Exception as e:
        print(f"Error loading font {font_path}: {e}. Falling back to default.")
        return ImageFont.load_default()

def wrap_text_by_word_count(text, font, max_width_pixels, max_words=None):
    """
    Wraps text to fit within a given pixel width and optionally truncates by word count,
    returning a list of lines.
    """
    if not text:
        return [""]
    words = text.split(' ')
    if max_words is not None and len(words) > max_words:
        words = words[:max_words]
        text_to_wrap = ' '.join(words) + "..."
    else:
        text_to_wrap = ' '.join(words)
    lines = []
    current_line_words = []
    dummy_img = Image.new('RGB', (1,1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    for word in text_to_wrap.split(' '):
        test_line = ' '.join(current_line_words + [word])
        text_bbox = dummy_draw.textbbox((0,0), test_line, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        if text_width <= max_width_pixels:
            current_line_words.append(word)
        else:
            if current_line_words:
                lines.append(' '.join(current_line_words))
            current_line_words = [word]
    if current_line_words:
        lines.append(' '.join(current_line_words))
    return lines

# --- Background Generator ---
class BackgroundGenerator:
    """Generates the gradient background for the post."""
    def generate_gradient_background(self, width, height, color1, color2):
        """Creates a diagonal gradient image."""
        img = Image.new('RGBA', (width, height), color1)
        draw = ImageDraw.Draw(img)
        for y in range(height):
            r1, g1, b1, a1 = color1
            r2, g2, b2, a2 = color2
            ratio_y = y / height
            r = int(r1 + (r2 - r1) * ratio_y)
            g = int(g1 + (g2 - g1) * ratio_y)
            b = int(b1 + (b2 - b1) * ratio_y)
            a = int(a1 + (a2 - a1) * ratio_y)
            draw.line([(0, y), (width, y)], fill=(r, g, b, a))
        return img


# --- API Callers ---

class NewsFetcher:
    """Fetches news and facts from various RSS feeds."""
    def _fetch_from_rss(self, rss_url, article_count=1, time_window_hours=48):
        """Fetches and parses articles from an RSS feed, filtering by recency."""
        try:
            feed = feedparser.parse(rss_url)
            if feed.bozo:
                print(f"Warning: RSS feed parsing issues for {rss_url}: {feed.bozo_exception}")
            recent_articles = []
            time_threshold = datetime.now(UTC) - timedelta(hours=time_window_hours)
            for entry in feed.entries:
                published_dt = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    published_dt = datetime(*entry.published_parsed[:6], tzinfo=UTC)
                elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                    published_dt = datetime(*entry.updated_parsed[:6], tzinfo=UTC)
                if not published_dt:
                    published_dt = datetime.now(UTC)
                if published_dt > time_threshold:
                    raw_description = entry.summary if hasattr(entry, 'summary') and entry.summary else (entry.title if hasattr(entry, 'title') else 'No Description')
                    clean_description = re.sub(r'<[^>]+>', '', raw_description).strip()
                    clean_description = re.sub(r'\s+', ' ', clean_description).strip()
                    recent_articles.append({
                        'title': entry.title if hasattr(entry, 'title') and entry.title else 'No Title',
                        'description': clean_description,
                        'url': entry.link if hasattr(entry, 'link') else rss_url,
                        'source': feed.feed.title if hasattr(feed.feed, 'title') and feed.feed.title else 'Unknown RSS',
                        'publishedAt': published_dt.isoformat() if published_dt else datetime.now(UTC).isoformat()
                    })
                    if len(recent_articles) >= article_count:
                        break
            return recent_articles
        except Exception as e:
            print(f"Error fetching from RSS feed {rss_url}: {e}")
            return []

    def get_single_content_item(self, content_type: str):
        """Fetches a single content item based on the specified type, using only RSS feeds."""
        world_news_sources = [
            {'type': 'rss', 'url': 'http://feeds.bbci.co.uk/news/world/rss.xml', 'name': 'BBC World News'},
            {'type': 'rss', 'url': 'https://www.reuters.com/rssfeed/worldNews', 'name': 'Reuters World News'},
        ]
        indian_news_sources = [
            {'type': 'rss', 'url': 'http://feeds.bbci.co.uk/news/world/asia/india/rss.xml', 'name': 'BBC India'},
            {'type': 'rss', 'url': 'https://www.thehindu.com/feeder/default.rss', 'name': 'The Hindu'},
        ]
        tech_news_sources = [
            {'type': 'rss', 'url': 'https://www.theverge.com/rss/index.xml', 'name': 'The Verge'},
            {'type': 'rss', 'url': 'https://techcrunch.com/feed/', 'name': 'TechCrunch'},
        ]
        environmental_news_sources = [
            {'type': 'rss', 'url': 'https://www.sciencedaily.com/rss/earth_climate.xml', 'name': 'ScienceDaily Environment'},
            {'type': 'rss', 'url': 'https://www.theguardian.com/environment/rss', 'name': 'The Guardian Environment'},
        ]
        content_item = None
        articles = []
        selected_sources = []
        if content_type == 'world_news':
            selected_sources = world_news_sources
        elif content_type == 'indian_news':
            selected_sources = indian_news_sources
        elif content_type == 'tech_news':
            selected_sources = tech_news_sources
        elif content_type == 'environmental_news':
            selected_sources = environmental_news_sources
        random.shuffle(selected_sources)
        for source_info in selected_sources:
            print(f"Fetching {content_type.replace('_', ' ').title()} from: {source_info['name']} ({source_info['type']})...")
            articles = self._fetch_from_rss(source_info['url'], article_count=1)
            if articles:
                break
        if articles:
            article = articles[0]
            source_name = article.get('source', f'Unknown {content_type.replace("_", " ").title()} Source')
            content_item = {
                'type': content_type,
                'title': article.get('title', 'No Title'),
                'description': article.get('description', article.get('title', 'No Description')),
                'url': article.get('url', ''),
                'source': source_name,
                'publishedAt': article.get('publishedAt', datetime.now(UTC).isoformat())
            }
        else:
            print(f"No recent RSS articles found for {content_type.replace('_', ' ').title()}.")
            return None
        return content_item


class TextProcessor:
    """Summarizes and enhances text using OpenRouter AI (DeepSeek model)."""
    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.model = OPENROUTER_MODEL
        self.site_url = OPENROUTER_SITE_URL
        self.site_name = OPENROUTER_SITE_NAME
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.api_key)

    def _call_ai_api(self, messages):
        """Helper to call the OpenRouter API. Returns (short_title, summary, success_flag)."""
        if not self.api_key or "YOUR_" in self.api_key:
            print("OPENROUTER_API_KEY is not set. Skipping AI text processing.")
            return "AI Key Error", "Please set your OpenRouter API key in config.py.", False
        try:
            completion = self.client.chat.completions.create(
                extra_headers={"HTTP-Referer": self.site_url, "X-Title": self.site_name},
                model=self.model, messages=messages, temperature=0.7, max_tokens=500,
                response_format={"type": "json_object"}
            )
            if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
                ai_content_str = completion.choices[0].message.content
                try:
                    parsed_data = json.loads(ai_content_str)
                    short_title = parsed_data.get('short_title', "Untitled")
                    summary = parsed_data.get('summary_text', "No summary available.")
                    title_words = short_title.split()
                    short_title = ' '.join(title_words[:TITLE_MAX_WORDS]) if len(title_words) > TITLE_MAX_WORDS else short_title
                    summary_words = summary.split()
                    if len(summary_words) > SUMMARY_MAX_WORDS:
                        summary = ' '.join(summary_words[:SUMMARY_MAX_WORDS])
                    return short_title, summary, True
                except json.JSONDecodeError:
                    print(f"Warning: OpenRouter did not return valid JSON. Raw response: {ai_content_str[:200]}...")
                    return "AI Title Error", "AI summary generation failed. Check API response.", False
            print(f"OpenRouter response missing expected content: {completion}")
            return "AI Response Error", "AI response structure invalid.", False
        except Exception as e:
            print(f"Error calling OpenRouter API: {e}")
            return "API Error", f"AI API request failed: {e}", False

    def process_text(self, title, description, post_type, style_recommendations=""):
        """Generates concise title and summary. Returns (short_title, summary, success_flag)."""
        messages = [
            {"role": "system", "content": f"You are an expert content summarizer for social media news posts. Your task is to generate a very concise headline and a clear, standalone summary from provided news content. The summary MUST be comprehensive enough to explain the main incident or topic clearly. Adhere strictly to the word and line count constraints for the output format. The output MUST always be a valid JSON object. Consider the following style recommendations: {style_recommendations}"},
            {"role": "user", "content": f"Generate a summary and title for the following content.\n1. The title MUST be exactly {TITLE_MAX_WORDS} words long.\n2. The summary MUST be between {SUMMARY_MIN_WORDS} and {SUMMARY_MAX_WORDS} words.\n3. The summary should be approximately 4-6 lines long when formatted for display on a social media image (assuming a width of {CANVAS_WIDTH - (LEFT_PADDING + RIGHT_PADDING)} pixels with font size {FONT_SIZE_SUMMARY}).\n4. The summary MUST be a complete thought and end gracefully.\n\nContent Type: {post_type.replace('_', ' ').title()}\nOriginal Title: {title}\nOriginal Description: {description}\n\nReturn ONLY the JSON object with two keys: \"short_title\" and \"summary_text\"."}
        ]
        short_title, summary, success = self._call_ai_api(messages)
        if not success:
            print("AI text processing failed. Using truncated original description as fallback.")
            fallback_summary = ' '.join(description.split()[:SUMMARY_MAX_WORDS])
            final_short_title = ' '.join(title.split()[:TITLE_MAX_WORDS])
            return final_short_title, fallback_summary, False
        return short_title, summary, True


class CaptionGenerator:
    """Generates Instagram captions, hashtags, and image prompts using OpenRouter AI (Mistral model)."""
    def __init__(self):
        self.api_key = OPENROUTER_MISTRAL_API_KEY
        self.model = OPENROUTER_MISTRAL_MODEL
        self.site_url = OPENROUTER_SITE_URL
        self.site_name = OPENROUTER_SITE_NAME
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.api_key)

    def generate_caption_and_hashtags(self, short_title, summary, style_recommendations=""):
        """Generates an Instagram-style caption and 10 relevant hashtags."""
        if not self.api_key or "YOUR_" in self.api_key:
            print("OPENROUTER_MISTRAL_API_KEY is not set. Skipping caption/hashtag generation.")
            return "Generated caption fallback.", ["#news", "#update"], False
        messages = [
            {"role": "system", "content": f"You are a creative social media manager specializing in Instagram posts for news. Your task is to generate a concise and engaging Instagram caption and exactly 10 trending hashtags based on a news title and summary. The output MUST be a valid JSON object with keys \"caption\" (string) and \"hashtags\" (array of strings). Consider the following style recommendations: {style_recommendations}"},
            {"role": "user", "content": f"Generate an Instagram caption and 10 relevant hashtags.\n\nNews Title: {short_title}\nNews Summary: {summary}\n\nReturn ONLY the JSON object with two keys: \"caption\" and \"hashtags\". Example: {{\"caption\": \"Example caption...\", \"hashtags\": [\"#example\", \"#trending\"]}}"}
        ]
        try:
            completion = self.client.chat.completions.create(
                extra_headers={"HTTP-Referer": self.site_url, "X-Title": self.site_name},
                model=self.model, messages=messages, temperature=0.7, max_tokens=200,
                response_format={"type": "json_object"}
            )
            if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
                ai_content_str = completion.choices[0].message.content
                try:
                    parsed_data = json.loads(ai_content_str)
                    caption = parsed_data.get('caption', "Engaging caption from Mistral.")
                    hashtags = parsed_data.get('hashtags', [])
                    if not isinstance(hashtags, list) or not all(isinstance(h, str) for h in hashtags):
                        hashtags = ["#news", "#update"]
                    if len(hashtags) > 10:
                        hashtags = hashtags[:10]
                    elif len(hashtags) < 10:
                        generic_hashtags = ["#dailynews", "#breaking", "#insightpulse", "#info", "#currentaffairs"]
                        for gen_tag in generic_hashtags:
                            if len(hashtags) >= 10: break
                            if gen_tag not in hashtags: hashtags.append(gen_tag)
                    hashtags = [h if h.startswith('#') else f'#{h}' for h in hashtags]
                    return caption, hashtags, True
                except json.JSONDecodeError:
                    print(f"Warning: Mistral did not return valid JSON for caption/hashtags. Raw: {ai_content_str[:200]}...")
                    return "Caption generation failed due to invalid JSON from AI.", ["#error", "#news"], False
            print(f"Mistral response missing expected content for caption/hashtags: {completion}")
            return "Caption generation failed: AI response structure invalid.", ["#error", "#news"], False
        except Exception as e:
            print(f"Error calling Mistral API for caption/hashtags: {e}")
            return f"Caption generation failed: {e}", ["#api_error", "#news"], False

    def generate_image_prompt(self, short_title, summary):
        """Generates a descriptive, text-free image prompt using Mistral."""
        if not self.api_key or "YOUR_" in self.api_key:
            print("OPENROUTER_MISTRAL_API_KEY is not set. Skipping image prompt generation.")
            return f"A symbolic image related to {short_title}", False
        messages = [
            {"role": "system", "content": "You are an expert in creating image generation prompts for diffusion models. Your task is to generate a single, descriptive, and photorealistic image prompt based on a news headline and summary. The prompt MUST NOT contain any words, text, letters, or signs. It should describe a visual scene that captures the essence and mood of the news story in a symbolic or representative way. Return ONLY the prompt string."},
            {"role": "user", "content": f"Generate a text-free, photorealistic image prompt for the following news content:\n\nNews Title: {short_title}\nNews Summary: {summary}\n\nReturn ONLY the prompt string."}
        ]
        try:
            completion = self.client.chat.completions.create(
                extra_headers={"HTTP-Referer": self.site_url, "X-Title": self.site_name},
                model=self.model, messages=messages, temperature=0.8, max_tokens=150,
            )
            if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
                image_prompt = completion.choices[0].message.content.strip().replace('"', '')
                print(f"Generated Image Prompt: {image_prompt}")
                return image_prompt, True
            else:
                print("Mistral response missing content for image prompt.")
                return f"A symbolic image related to {short_title}", False
        except Exception as e:
            print(f"Error calling Mistral API for image prompt generation: {e}")
            return f"A symbolic image related to {short_title}", False


class FluxImageGenerator:
    """Generates images by downloading the FLUX.1-schnell model from Hugging Face."""
    def __init__(self):
        self.pipe = None
        if not FLUX_LIBRARIES_AVAILABLE:
            print("FLUX dependencies not installed. FLUX Image Generator is disabled.")
            return

        if not HUGGING_FACE_TOKEN:
            print("-" * 80)
            print("CRITICAL: HUGGING_FACE_TOKEN environment variable is not set.")
            print("Please add a GitHub Secret named HF_TOKEN with your Hugging Face read access token.")
            print("Image generation will fall back to stock photo APIs.")
            print("-" * 80)
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing FLUX model on device: {self.device}")
        try:
            torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            self.pipe = FluxPipeline.from_pretrained(
                FLUX_MODEL_ID,
                torch_dtype=torch_dtype,
                token=HUGGING_FACE_TOKEN  # Use the token for authentication
            )
            self.pipe.enable_model_cpu_offload()
            print("FLUX.1-schnell model loaded successfully using token.")
        except Exception as e:
            print(f"CRITICAL: Failed to load FLUX.1-schnell model: {e}")
            print("Please ensure you have accepted the terms on the model page and the token is correct.")
            self.pipe = None

    def generate_image(self, prompt):
        """Generates a single image from a text prompt."""
        if not self.pipe:
            print("FLUX model not available. Skipping generation.")
            return None
        try:
            print(f"Generating image with FLUX model for prompt: {prompt[:80]}...")
            generator = torch.Generator(device="cpu").manual_seed(random.randint(0, 2**32 - 1))
            image = self.pipe(
                prompt=prompt, guidance_scale=0.0, num_inference_steps=4,
                max_sequence_length=256, generator=generator
            ).images[0]
            print("FLUX image generated successfully.")
            return image
        except Exception as e:
            print(f"Error during FLUX image generation: {e}")
            return None


class ImageFetcher:
    """Fetches images from Pexels as a fallback."""
    def __init__(self):
        self.pexels_api_key = PEXELS_API_KEY
        self.pexels_api_url = PEXELS_API_URL

    def fetch_image(self, prompt, width=IMAGE_DISPLAY_WIDTH, height=IMAGE_DISPLAY_HEIGHT):
        """Attempts to fetch an image from Pexels."""
        try:
            if not self.pexels_api_key or "YOUR_" in self.pexels_api_key:
                print("PEXELS_API_KEY is not set. Skipping Pexels.")
                return None

            headers = {"Authorization": self.pexels_api_key}
            params = {"query": prompt, "orientation": "portrait", "size": "large", "per_page": 1}
            print(f"Searching Pexels for image with prompt: {prompt[:50]}...")
            response = requests.get(f"{self.pexels_api_url}/search", headers=headers, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            if data and data['photos']:
                image_url = data['photos'][0]['src']['original']
                img_data = requests.get(image_url, stream=True, timeout=15)
                img_data.raise_for_status()
                return Image.open(io.BytesIO(img_data.content))
            return None
        except Exception as e:
            print(f"Error fetching from Pexels: {e}")
            return None


class ImageLocalProcessor:
    """Handles local image processing and text overlays."""
    def _get_filtered_source_display(self, original_source_text: str) -> str:
        """Filters and shortens the source text for display."""
        if not original_source_text: return "UNKNOWN"
        source_map = {
            "BBC WORLD NEWS": "BBC", "BBC INDIA": "BBC", "REUTERS WORLD NEWS": "REUTERS",
            "THE HINDU": "HINDU", "THE VERGE": "THE VERGE", "TECHCRUNCH": "TECHCRUNCH",
            "SCIENCEDAILY ENVIRONMENT": "SCIENCEDAILY", "THE GUARDIAN ENVIRONMENT": "GUARDIAN ENV"
        }
        mapped_source = source_map.get(original_source_text.upper().strip())
        if mapped_source: return mapped_source
        processed_source = original_source_text.upper().strip()
        noisy_phrases = [r'\bNEWS\b', r'\bREPORTS\b', r'\.COM']
        for phrase in noisy_phrases:
            processed_source = re.sub(phrase, '', processed_source).strip()
        processed_source = re.sub(r'\s+', ' ', processed_source).strip()
        words = processed_source.split()
        return ' '.join(words[:3]) if words else "UNKNOWN"

    def overlay_text(self, base_pil_image, post_data):
        """Creates the final post image with gradient background, central image, and text overlays."""
        try:
            background_gen = BackgroundGenerator()
            final_canvas = background_gen.generate_gradient_background(CANVAS_WIDTH, CANVAS_HEIGHT, COLOR_GRADIENT_TOP_LEFT, COLOR_GRADIENT_BOTTOM_RIGHT)
            draw = ImageDraw.Draw(final_canvas)
            dummy_draw_for_text_bbox = ImageDraw.Draw(Image.new('RGB', (1,1)))
            content_type_display = post_data.get('type', 'NEWS').upper().replace('_', ' ')
            font_top_left_text = load_font(FONT_PATH_BOLD, FONT_SIZE_TOP_LEFT_TEXT)
            draw.text((TOP_LEFT_TEXT_POS_X, TOP_LEFT_TEXT_POS_Y), content_type_display, font=font_top_left_text, fill=COLOR_LIGHT_GRAY_TEXT)
            timestamp_text = datetime.now().strftime("%d/%m/%Y %H:%M")
            font_timestamp = load_font(FONT_PATH_BOLD, FONT_SIZE_TIMESTAMP)
            timestamp_bbox = dummy_draw_for_text_bbox.textbbox((0,0), timestamp_text, font=font_timestamp)
            timestamp_width = timestamp_bbox[2] - timestamp_bbox[0]
            draw.text((TIMESTAMP_POS_X_RIGHT_ALIGN - timestamp_width, TIMESTAMP_POS_Y), timestamp_text, font=font_timestamp, fill=COLOR_WHITE)
            image_start_y = max(TOP_LEFT_TEXT_POS_Y + 35, TIMESTAMP_POS_Y + 32) + IMAGE_TOP_MARGIN_FROM_TOP_ELEMENTS
            news_image_for_display = ImageOps.fit(base_pil_image, (IMAGE_DISPLAY_WIDTH, IMAGE_DISPLAY_HEIGHT), Image.Resampling.LANCZOS)
            mask = Image.new('L', (IMAGE_DISPLAY_WIDTH, IMAGE_DISPLAY_HEIGHT), 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.rounded_rectangle((0, 0, IMAGE_DISPLAY_WIDTH, IMAGE_DISPLAY_HEIGHT), radius=IMAGE_ROUND_RADIUS, fill=255)
            news_image_x = (CANVAS_WIDTH - IMAGE_DISPLAY_WIDTH) // 2
            final_canvas.paste(news_image_for_display, (news_image_x, int(image_start_y)), mask)
            title_text_raw = str(post_data.get('title', 'NO TITLE')).upper()
            font_headline = load_font(FONT_PATH_EXTRABOLD, FONT_SIZE_HEADLINE)
            wrapped_title_lines = wrap_text_by_word_count(title_text_raw, font_headline, CANVAS_WIDTH - (LEFT_PADDING + RIGHT_PADDING), max_words=TITLE_MAX_WORDS)
            current_y_title = image_start_y + IMAGE_DISPLAY_HEIGHT + TITLE_TOP_MARGIN_FROM_IMAGE
            for line in wrapped_title_lines:
                line_bbox = dummy_draw_for_text_bbox.textbbox((0,0), line, font=font_headline)
                line_width = line_bbox[2] - line_bbox[0]
                text_x_centered = (CANVAS_WIDTH - line_width) / 2
                draw.text((text_x_centered, current_y_title), line, font=font_headline, fill=COLOR_WHITE)
                current_y_title += (line_bbox[3] - line_bbox[1]) + TITLE_LINE_SPACING
            summary_text_raw = str(post_data.get('summary', 'No summary provided.'))
            font_summary = load_font(FONT_PATH_REGULAR, FONT_SIZE_SUMMARY)
            wrapped_summary_lines = wrap_text_by_word_count(summary_text_raw, font_summary, CANVAS_WIDTH - (LEFT_PADDING + RIGHT_PADDING), max_words=SUMMARY_MAX_WORDS)
            current_y_summary = current_y_title + SUMMARY_TOP_MARGIN_FROM_TITLE
            for line in wrapped_summary_lines:
                draw.text((LEFT_PADDING, current_y_summary), line, font=font_summary, fill=COLOR_WHITE)
                line_height = (dummy_draw_for_text_bbox.textbbox((0,0), line, font=font_summary)[3] - dummy_draw_for_text_bbox.textbbox((0,0), line, font=font_summary)[1])
                current_y_summary += line_height + SUMMARY_LINE_SPACING
            current_y_summary -= SUMMARY_LINE_SPACING
            divider_y = current_y_summary + DIVIDER_Y_OFFSET_FROM_SUMMARY
            color_red_alpha = (COLOR_RED[0], COLOR_RED[1], COLOR_RED[2], int(255 * 0.3))
            draw.line([(LEFT_PADDING, divider_y), (CANVAS_WIDTH - RIGHT_PADDING, divider_y)], fill=color_red_alpha, width=DIVIDER_LINE_THICKNESS)
            filtered_source_for_display = self._get_filtered_source_display(post_data.get('source', 'SOURCE'))
            source_text_display = f"Source - {filtered_source_for_display}"
            font_source = load_font(FONT_PATH_REGULAR, FONT_SIZE_SOURCE)
            source_bbox_content = dummy_draw_for_text_bbox.textbbox((0,0), source_text_display, font=font_source)
            source_width_content = source_bbox_content[2] - source_bbox_content[0]
            source_height_content = source_bbox_content[3] - source_bbox_content[1]
            source_rect_width = source_width_content + (2 * SOURCE_RECT_PADDING_X)
            source_rect_height = source_height_content + (2 * SOURCE_RECT_PADDING_Y)
            source_rect_y2 = CANVAS_HEIGHT - BOTTOM_PADDING
            source_rect_y1 = source_rect_y2 - source_rect_height
            source_text_y = source_rect_y1 + SOURCE_RECT_PADDING_Y
            logo_y = source_rect_y1 + (source_rect_height - LOGO_HEIGHT) / 2
            try:
                logo_image = Image.open(LOGO_PATH).convert("RGBA")
                logo_image.thumbnail((LOGO_WIDTH, LOGO_HEIGHT), Image.Resampling.LANCZOS)
                final_canvas.paste(logo_image, (LEFT_PADDING, int(logo_y)), logo_image)
            except Exception as e:
                print(f"Error embedding logo: {e}. Embedding text fallback.")
                draw.text((LEFT_PADDING, int(logo_y) + (LOGO_HEIGHT - 30) // 2), "Insight Pulse", font=load_font(FONT_PATH_BOLD, 30), fill=COLOR_WHITE)
            source_rect_x1 = SOURCE_POS_X_RIGHT_ALIGN - source_rect_width
            source_rect_x2 = SOURCE_POS_X_RIGHT_ALIGN
            draw.rounded_rectangle((source_rect_x1, source_rect_y1, source_rect_x2, source_rect_y2), radius=SOURCE_RECT_RADIUS, fill=COLOR_DARK_GRAY)
            draw.text((source_rect_x1 + SOURCE_RECT_PADDING_X, int(source_text_y)), source_text_display, font=font_source, fill=COLOR_WHITE)
            return final_canvas
        except Exception as e:
            print(f"Error during image overlay and composition: {e}")
            img_error = Image.new('RGB', (CANVAS_WIDTH, CANVAS_HEIGHT), color = (50, 50, 50))
            draw_error = ImageDraw.Draw(img_error)
            draw_error.text((50, CANVAS_HEIGHT // 2 - 20), "IMAGE COMPOSITION ERROR", font=load_font(FONT_PATH_REGULAR, 40), fill=(255,0,0))
            return img_error


class CloudinaryUploader:
    """Handles uploading images to Cloudinary."""
    def __init__(self):
        cloudinary.config(cloud_name=CLOUDINARY_CLOUD_NAME, api_key=CLOUDINARY_API_KEY, api_secret=CLOUDINARY_API_SECRET)

    def upload_image(self, image_path, public_id, folder="news_posts"):
        """Uploads an image to Cloudinary."""
        try:
            if not CLOUDINARY_CLOUD_NAME or "YOUR_" in CLOUDINARY_CLOUD_NAME:
                print("Cloudinary credentials are not set. Skipping upload.")
                return None
            print(f"Uploading {image_path} to Cloudinary folder '{folder}' with public_id '{public_id}'...")
            upload_result = cloudinary.uploader.upload(image_path, public_id=public_id, folder=folder)
            secure_url = upload_result.get('secure_url')
            if secure_url:
                print(f"Image uploaded to Cloudinary: {secure_url}")
                return secure_url
            else:
                print(f"Cloudinary upload failed: No secure_url in response.")
                return None
        except Exception as e:
            print(f"Error uploading image to Cloudinary: {e}")
            return None


class InstagramPoster:
    """Handles posting images to Instagram via the Facebook Graph API."""
    def __init__(self):
        self.access_token = FB_PAGE_ACCESS_TOKEN
        self.instagram_business_account_id = INSTAGRAM_BUSINESS_ACCOUNT_ID
        self.graph_api_base_url = "https://graph.facebook.com/v19.0/"

    def post_image(self, image_url, caption):
        """Posts an image to Instagram."""
        if not self.access_token or "YOUR_" in self.access_token or not self.instagram_business_account_id or "YOUR_" in self.instagram_business_account_id:
            print("Instagram credentials not set. Skipping post.")
            return False
        if not image_url:
            print("No image URL provided for Instagram post. Skipping.")
            return False
        print(f"Attempting to post to Instagram (Account ID: {self.instagram_business_account_id})...")
        media_container_url = f"{self.graph_api_base_url}{self.instagram_business_account_id}/media"
        media_params = {'image_url': image_url, 'caption': caption, 'access_token': self.access_token}
        try:
            response = requests.post(media_container_url, params=media_params, timeout=30)
            response.raise_for_status()
            media_container_id = response.json().get('id')
            if not media_container_id:
                print("Failed to get media container ID.")
                return False
            publish_url = f"{self.graph_api_base_url}{self.instagram_business_account_id}/media_publish"
            publish_params = {'creation_id': media_container_id, 'access_token': self.access_token}
            response = requests.post(publish_url, params=publish_params, timeout=30)
            response.raise_for_status()
            post_id = response.json().get('id')
            if post_id:
                print(f"Post successfully published to Instagram with ID: {post_id}")
                return True
            else:
                print(f"Instagram publish failed: No post ID in response.")
                return False
        except Exception as e:
            print(f"Error posting to Instagram: {e}")
            return False


class LocalSaver:
    """Saves data and images locally to JSON and Excel."""
    def __init__(self, image_output_dir, json_output_dir, excel_output_dir, all_posts_json_file, all_posts_excel_file):
        self.IMAGE_OUTPUT_DIR = image_output_dir
        self.JSON_OUTPUT_DIR = json_output_dir
        self.EXCEL_OUTPUT_DIR = excel_output_dir
        self.ALL_POSTS_JSON_FILE = all_posts_json_file
        self.ALL_POSTS_EXCEL_FILE = all_posts_excel_file
        os.makedirs(self.IMAGE_OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.JSON_OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.EXCEL_OUTPUT_DIR, exist_ok=True)

    def save_post(self, post_data):
        """Saves a single post's data and image."""
        post_type_label = post_data.get('type', 'post').replace('_', '-')
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        post_id = f"{post_type_label}_{timestamp_str}_post-{workflow_manager.get_current_post_number()}"
        post_data['Post_ID'] = post_id
        image_filename = f"{post_id}.png"
        image_path = os.path.join(self.IMAGE_OUTPUT_DIR, image_filename)
        if 'final_image' in post_data and isinstance(post_data['final_image'], Image.Image):
            try:
                post_data['final_image'].save(image_path)
                print(f"Image saved to: {image_path}")
            except Exception as e:
                print(f"Error saving image to {image_path}: {e}")
                image_path = "Error saving image"
        else:
            image_path = "No image generated/saved"
        metadata = {
            "Post_ID": post_data['Post_ID'], "Title": post_data.get('title'), "Summary": post_data.get('summary'),
            "SEO_Caption": post_data.get('seo_caption'), "Hashtags": ', '.join(post_data.get('hashtags', [])),
            "Local_Image_Path": image_path, "Cloudinary_URL": post_data.get('cloudinary_url', 'N/A'),
            "Instagram_Posted": post_data.get('instagram_posted', False), "Timestamp": datetime.now(UTC).isoformat(),
            "Source_Type": post_data.get('type'), "Source_URL": post_data.get('url', ''),
            "Original_Source": post_data.get('source', 'N/A'), "Original_Description": post_data.get('original_description', 'N/A'),
            "Image_Status": post_data.get('image_status', 'unknown')
        }
        try:
            existing_data = []
            if os.path.exists(self.ALL_POSTS_JSON_FILE):
                with open(self.ALL_POSTS_JSON_FILE, 'r', encoding='utf-8') as f:
                    try: existing_data = json.load(f)
                    except json.JSONDecodeError: existing_data = []
            existing_data.append(metadata)
            with open(self.ALL_POSTS_JSON_FILE, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=4)
        except Exception as e:
            print(f"Error saving to JSON file {self.ALL_POSTS_JSON_FILE}: {e}")
        try:
            df = pd.DataFrame([metadata])
            if os.path.exists(self.ALL_POSTS_EXCEL_FILE):
                with pd.ExcelWriter(self.ALL_POSTS_EXCEL_FILE, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                    startrow = writer.sheets['Posts'].max_row if 'Posts' in writer.sheets else 0
                    df.to_excel(writer, sheet_name='Posts', index=False, header=(startrow == 0), startrow=startrow)
            else:
                df.to_excel(self.ALL_POSTS_EXCEL_FILE, sheet_name='Posts', index=False)
        except Exception as e:
            print(f"Error saving to Excel file {self.ALL_POSTS_EXCEL_FILE}: {e}")

    def load_all_posts_data(self):
        """Loads all historical post data from the JSON file."""
        if os.path.exists(self.ALL_POSTS_JSON_FILE):
            try:
                with open(self.ALL_POSTS_JSON_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                print(f"Error loading {self.ALL_POSTS_JSON_FILE}: {e}. Returning empty list.")
                return []
        return []


def _load_style_recommendations():
    """Loads the last saved style recommendations from a JSON file."""
    if os.path.exists(STYLE_RECOMMENDATIONS_FILE):
        try:
            with open(STYLE_RECOMMENDATIONS_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error loading style recommendations: {e}. Returning empty.")
            return {}
    return {}

def _save_style_recommendations(recommendations):
    """Saves the current style recommendations to a JSON file."""
    os.makedirs(os.path.dirname(STYLE_RECOMMENDATIONS_FILE), exist_ok=True)
    try:
        with open(STYLE_RECOMMENDATIONS_FILE, 'w') as f:
            json.dump(recommendations, f, indent=4)
    except Exception as e:
        print(f"Error saving style recommendations: {e}")

def perform_weekly_analysis(mistral_client, local_saver_instance):
    """Analyzes past week's content and generates style recommendations."""
    print("\n--- Performing Weekly Performance Analysis ---")
    all_posts_data = local_saver_instance.load_all_posts_data()
    one_week_ago = datetime.now(UTC) - timedelta(days=WEEKLY_ANALYSIS_INTERVAL_DAYS)
    past_week_posts = [p for p in all_posts_data if 'Timestamp' in p and datetime.fromisoformat(p['Timestamp'].replace("Z", "+00:00")) >= one_week_ago]

    if not past_week_posts:
        print("No posts found from the last week to analyze.")
        return {}

    past_content_summary = "\n".join([f"Post ({p['Timestamp']}):\n  Type: {p.get('Source_Type', 'N/A')}\n  Caption: {p.get('SEO_Caption', 'N/A')}\n  Hashtags: {p.get('Hashtags', 'N/A')}" for p in past_week_posts])
    system_message = "You are an expert social media content strategist. Analyze past content and provide actionable, creative recommendations to improve future content's reach and engagement. Be specific."
    prompt = f"Analyze the following past week's content. Provide specific recommendations to improve caption style, hashtag strategy, and content themes.\n\n{past_content_summary}"

    try:
        response = mistral_client.chat.completions.create(
            model=OPENROUTER_MISTRAL_MODEL,
            messages=[{"role": "system", "content": system_message}, {"role": "user", "content": prompt}],
            extra_headers={"HTTP-Referer": OPENROUTER_SITE_URL, "X-Site-Name": OPENROUTER_SITE_NAME},
        )
        recommendations_text = response.choices[0].message.content.strip()
        if recommendations_text:
            print("Mistral Analysis Result:\n", recommendations_text)
            new_recommendations = {"weekly_analysis": recommendations_text, "timestamp": datetime.now(UTC).isoformat()}
            _save_style_recommendations(new_recommendations)
            return new_recommendations
        return {}
    except Exception as e:
        print(f"Error calling Mistral for weekly analysis: {e}")
        return {}

def check_api_keys():
    """Checks if essential API keys are set."""
    warnings = []
    if not OPENROUTER_API_KEY or "YOUR_" in OPENROUTER_API_KEY:
        warnings.append("OPENROUTER_API_KEY is not set.")
    if not PEXELS_API_KEY or "YOUR_" in PEXELS_API_KEY:
        warnings.append("PEXELS_API_KEY is not set.")
    if not CLOUDINARY_CLOUD_NAME or "YOUR_" in CLOUDINARY_CLOUD_NAME:
        warnings.append("Cloudinary credentials are not set.")
    if not FB_PAGE_ACCESS_TOKEN or "YOUR_" in FB_PAGE_ACCESS_TOKEN:
        warnings.append("Instagram credentials are not set.")
    if ENABLE_FLUX_IMAGE_GENERATION and (not HUGGING_FACE_TOKEN or "YOUR_" in HUGGING_FACE_TOKEN):
        warnings.append("HUGGING_FACE_TOKEN is not set for FLUX model.")

    if warnings:
        print("\n--- API KEY WARNINGS ---")
        for warning in warnings:
            print(f"- {warning}")
        print("------------------------\n")
        # input("Press Enter to continue (or Ctrl+C to exit)...") # Pausing can be an issue in CI/CD

# --- Main Workflow Execution ---
if __name__ == "__main__":
    check_api_keys()

    workflow_manager = WorkflowStateManager()
    news_fetcher = NewsFetcher()
    text_processor = TextProcessor()
    image_fetcher = ImageFetcher()
    image_local_processor = ImageLocalProcessor()
    caption_generator = CaptionGenerator()
    cloudinary_uploader = CloudinaryUploader()
    instagram_poster = InstagramPoster()
    local_saver = LocalSaver(IMAGE_OUTPUT_DIR, JSON_OUTPUT_DIR, EXCEL_OUTPUT_DIR, ALL_POSTS_JSON_FILE, ALL_POSTS_EXCEL_FILE)
    mistral_client_for_analysis = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_MISTRAL_API_KEY)

    flux_image_gen = None
    if ENABLE_FLUX_IMAGE_GENERATION and FLUX_LIBRARIES_AVAILABLE:
        flux_image_gen = FluxImageGenerator()

    try:
        recommendation_text_for_llm = _load_style_recommendations().get('weekly_analysis', '')
        if workflow_manager.should_run_weekly_analysis():
            new_recs = perform_weekly_analysis(mistral_client_for_analysis, local_saver)
            if new_recs:
                recommendation_text_for_llm = new_recs.get('weekly_analysis', '')
            workflow_manager.update_last_analysis_timestamp()

        content_type = workflow_manager.get_current_post_type()
        post_num = workflow_manager.get_current_post_number()
        print(f"\n--- Processing Post {post_num}/{len(CONTENT_TYPE_CYCLE)} (Type: {content_type.replace('_', ' ').title()}) ---")

        post_data = news_fetcher.get_single_content_item(content_type)
        if not post_data:
            print(f"No content for '{content_type}'. Skipping.")
            workflow_manager.increment_post_type_index()
            sys.exit(0)

        post_data['original_description'] = post_data.get('description', 'N/A')
        title, summary, success = text_processor.process_text(post_data['title'], post_data['description'], post_data['type'], recommendation_text_for_llm)
        if not success:
            print("Text processing failed. Skipping.")
            workflow_manager.increment_post_type_index()
            sys.exit(0)
        
        post_data['title'], post_data['summary'] = title, summary
        print(f"Generated Title: {title}\nGenerated Summary: {summary}")

        pil_image = None
        if flux_image_gen and flux_image_gen.pipe:
            img_prompt, prompt_ok = caption_generator.generate_image_prompt(title, summary)
            if prompt_ok:
                pil_image = flux_image_gen.generate_image(img_prompt)
                post_data['image_status'] = 'generated_flux' if pil_image else 'flux_failed'
        
        if not pil_image:
            print("Falling back to stock photo APIs.")
            api_prompt = f"{title} {summary}"
            pil_image = image_fetcher.fetch_image(api_prompt)
            if pil_image:
                post_data['image_status'] = 'fetched_api'

        if not pil_image:
            print("All image sources failed. Generating placeholder.")
            pil_image = Image.new('RGB', (IMAGE_DISPLAY_WIDTH, IMAGE_DISPLAY_HEIGHT), color=(20, 20, 20))
            draw = ImageDraw.Draw(pil_image)
            font = load_font(FONT_PATH_REGULAR, 40)
            draw.text((50, 50), "Image Not Available", font=font, fill=(200, 200, 200))
            post_data['image_status'] = 'placeholder'
        
        post_data['final_image'] = image_local_processor.overlay_text(pil_image, post_data)
        
        caption, hashtags, _ = caption_generator.generate_caption_and_hashtags(title, summary, recommendation_text_for_llm)
        post_data['seo_caption'], post_data['hashtags'] = caption, hashtags

        local_saver.save_post(post_data)
        
        img_path = os.path.join(IMAGE_OUTPUT_DIR, f"{post_data['Post_ID']}.png")
        cloudinary_url = None
        if os.path.exists(img_path):
            cloudinary_url = cloudinary_uploader.upload_image(img_path, post_data['Post_ID'], "insight_pulse_posts")
        post_data['cloudinary_url'] = cloudinary_url

        if cloudinary_url:
            full_caption = f"{caption}\n\n{' '.join(hashtags)}"
            post_data['instagram_posted'] = instagram_poster.post_image(cloudinary_url, full_caption)
        else:
            post_data['instagram_posted'] = False

        workflow_manager.increment_post_type_index()
        print(f"Successfully processed post {post_num}/{len(CONTENT_TYPE_CYCLE)}. State updated.")

    except Exception as e:
        print(f"\nAn unhandled error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)
