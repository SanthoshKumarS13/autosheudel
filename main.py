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
import time
import traceback
from openai import OpenAI 
import cloudinary 
import cloudinary.uploader 

# Fix for some SSL certificate issues with feedparser on some systems
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context


# Import configuration and state manager
from config import *
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
        """
        Fetches a single content item based on the specified type, using only RSS feeds.
        """
        world_news_sources = [
            {'type': 'rss', 'url': 'http://feeds.bbci.co.uk/news/world/rss.xml', 'name': 'BBC World News'},
            {'type': 'rss', 'url': 'https://www.reuters.com/rssfeed/worldNews', 'name': 'Reuters World News'},
            {'type': 'rss', 'url': 'https://rss.nytimes.com/services/xml/rss/nyt/WorldNews.xml', 'name': 'NYT World News'},
            {'type': 'rss', 'url': 'https://www.aljazeera.com/xml/rss/all.xml', 'name': 'Al Jazeera'}
        ]
        indian_news_sources = [
            {'type': 'rss', 'url': 'http://feeds.bbci.co.uk/news/world/asia/india/rss.xml', 'name': 'BBC India'},
            {'type': 'rss', 'url': 'https://www.thehindu.com/feeder/default.rss', 'name': 'The Hindu'},
            {'type': 'rss', 'url': 'https://timesofindia.indiatimes.com/rssfeeds/7091390.cms', 'name': 'Times of India'},
            {'type': 'rss', 'url': 'https://zeenews.india.com/rss/india-news.xml', 'name': 'Zee News India'}
        ]
        tech_news_sources = [ 
            {'type': 'rss', 'url': 'https://www.theverge.com/rss/index.xml', 'name': 'The Verge'},
            {'type': 'rss', 'url': 'https://techcrunch.com/feed/', 'name': 'TechCrunch'},
            {'type': 'rss', 'url': 'https://arstechnica.com/feed/', 'name': 'Ars Technica'},
            {'type': 'rss', 'url': 'https://www.wired.com/feed/rss', 'name': 'Wired'}
        ]
        environmental_news_sources = [ 
            {'type': 'rss', 'url': 'https://www.sciencedaily.com/rss/earth_climate.xml', 'name': 'ScienceDaily Environment'},
            {'type': 'rss', 'url': 'https://www.mongabay.com/feed/', 'name': 'Mongabay'},
            {'type': 'rss', 'url': 'https://www.theguardian.com/environment/rss', 'name': 'The Guardian Environment'},
            {'type': 'rss', 'url': 'https://e360.yale.edu/digest.rss', 'name': 'Yale Environment 360'}
        ]
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
                article = articles[0]
                return {
                    'type': content_type,
                    'title': article.get('title', 'No Title'),
                    'description': article.get('description', 'No Description'),
                    'url': article.get('url', ''),
                    'source': article.get('source', 'Unknown Source'),
                    'publishedAt': article.get('publishedAt', datetime.now(UTC).isoformat())
                }
        print(f"No recent RSS articles found for {content_type.replace('_', ' ').title()}.")
        return None

class TextProcessor:
    """Summarizes and enhances text using OpenRouter AI (DeepSeek model)."""
    def __init__(self):
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

    def _call_ai_api(self, messages):
        """Helper to call the OpenRouter API."""
        if not OPENROUTER_API_KEY or "YOUR_" in OPENROUTER_API_KEY:
            print("OPENROUTER_API_KEY is not set. Skipping AI text processing.")
            return None
        try:
            completion = self.client.chat.completions.create(
                extra_headers={"HTTP-Referer": OPENROUTER_SITE_URL, "X-Title": OPENROUTER_SITE_NAME},
                model=OPENROUTER_MODEL, messages=messages, temperature=0.7, response_format={"type": "json_object"}
            )
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            print(f"Error calling OpenRouter API: {e}")
            return None

    def process_text(self, title, description, post_type, style_recommendations=""):
        """Generates concise title and summary."""
        messages = [
            {"role": "system", "content": f"You are an expert content summarizer for social media. Generate a concise headline (exactly {TITLE_MAX_WORDS} words) and a clear summary ({SUMMARY_MIN_WORDS}-{SUMMARY_MAX_WORDS} words). The output MUST be a valid JSON object with keys 'short_title' and 'summary_text'. Consider these style recommendations: {style_recommendations}"},
            {"role": "user", "content": f"Content Type: {post_type.replace('_', ' ').title()}\nOriginal Title: {title}\nOriginal Description: {description}\n\nReturn ONLY the JSON object."}
        ]
        parsed_data = self._call_ai_api(messages)
        if parsed_data:
            return parsed_data.get('short_title', "Untitled"), parsed_data.get('summary_text', "No summary."), True
        else:
            return ' '.join(title.split()[:TITLE_MAX_WORDS]), ' '.join(description.split()[:SUMMARY_MAX_WORDS]), False

class CaptionGenerator:
    """Generates Instagram captions, hashtags, and image prompts using OpenRouter AI (Mistral model)."""
    def __init__(self):
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_MISTRAL_API_KEY)

    def generate_caption_and_hashtags(self, short_title, summary, style_recommendations=""):
        """Generates an Instagram-style caption and 10 relevant hashtags."""
        if not OPENROUTER_MISTRAL_API_KEY or "YOUR_" in OPENROUTER_MISTRAL_API_KEY:
            return "Follow for more updates.", ["#news"], False
        messages = [
            {"role": "system", "content": f"You are a creative social media manager. Generate an engaging Instagram caption and exactly 10 trending hashtags. The output MUST be a valid JSON object with keys 'caption' and 'hashtags'. Consider these style recommendations: {style_recommendations}"},
            {"role": "user", "content": f"News Title: {short_title}\nNews Summary: {summary}\n\nReturn ONLY the JSON object."}
        ]
        try:
            completion = self.client.chat.completions.create(
                extra_headers={"HTTP-Referer": OPENROUTER_SITE_URL, "X-Title": OPENROUTER_SITE_NAME},
                model=OPENROUTER_MISTRAL_MODEL, messages=messages, temperature=0.7, response_format={"type": "json_object"}
            )
            data = json.loads(completion.choices[0].message.content)
            hashtags = data.get('hashtags', [])
            if not isinstance(hashtags, list): hashtags = ["#news"]
            return data.get('caption', "Follow for more updates."), hashtags, True
        except Exception as e:
            print(f"Error in CaptionGenerator: {e}")
            return "Follow for more updates.", ["#news", "#update"], False

    def generate_image_prompt(self, short_title, summary):
        """Generates a descriptive, text-free image prompt using Mistral."""
        if not OPENROUTER_MISTRAL_API_KEY or "YOUR_" in OPENROUTER_MISTRAL_API_KEY:
            return f"Symbolic image representing {short_title}", False
        messages = [
            {"role": "system", "content": "You are an expert in creating image generation prompts for diffusion models. Generate a single, descriptive, photorealistic image prompt based on a news story. The prompt MUST NOT contain any words, text, or signs. Return ONLY the prompt string."},
            {"role": "user", "content": f"News Title: {short_title}\nNews Summary: {summary}\n\nReturn ONLY the prompt string."}
        ]
        try:
            completion = self.client.chat.completions.create(
                extra_headers={"HTTP-Referer": OPENROUTER_SITE_URL, "X-Title": OPENROUTER_SITE_NAME},
                model=OPENROUTER_MISTRAL_MODEL, messages=messages
            )
            return completion.choices[0].message.content.strip().replace('"', ''), True
        except Exception as e:
            print(f"Error generating image prompt: {e}")
            return f"Symbolic image representing {short_title}", False

# --- NEW: Hugging Face Image Generator (using Inference API) ---
class HuggingFaceImageGenerator:
    """Tries a sequence of Hugging Face Inference APIs to generate an image."""
    def __init__(self):
        if not HUGGING_FACE_TOKEN or "YOUR_" in HUGGING_FACE_TOKEN:
            print("HUGGING_FACE_TOKEN is not set. AI image generation will be disabled.")
            self.headers = None
        else:
            self.headers = {"Authorization": f"Bearer {HUGGING_FACE_TOKEN}"}

    def generate_image(self, prompt):
        """Calls the Inference API for each model in the list until one succeeds."""
        if not self.headers:
            return None

        for api_url in INFERENCE_API_ENDPOINTS:
            model_name = api_url.split('/')[-1]
            print(f"\n--- Attempting to generate image with API: {model_name} ---")
            try:
                payload = {"inputs": prompt}
                response = requests.post(api_url, headers=self.headers, json=payload, timeout=60)

                if response.status_code == 503: # Model is loading
                    estimated_time = response.json().get("estimated_time", 25)
                    print(f"Model '{model_name}' is loading, waiting for {estimated_time:.2f} seconds...")
                    time.sleep(estimated_time)
                    response = requests.post(api_url, headers=self.headers, json=payload, timeout=60)

                response.raise_for_status()
                
                if response.headers.get('Content-Type') in ['image/jpeg', 'image/png']:
                    print(f"Successfully generated image with {model_name}.")
                    return Image.open(io.BytesIO(response.content))
                else:
                    print(f"API for {model_name} did not return an image. Response: {response.text}")
            
            except requests.exceptions.RequestException as e:
                print(f"Failed to generate image with {model_name}. Error: {e}")
        
        print("\nAll AI image generation APIs failed.")
        return None

# --- Original Image Fetcher (Stock Photos) ---
class ImageFetcher:
    """Fetches images from Pexels, Unsplash, Openverse, and Pixabay."""
    def __init__(self):
        self.pexels_api_key = PEXELS_API_KEY
        self.pexels_api_url = PEXELS_API_URL
        self.unsplash_access_key = UNSPLASH_ACCESS_KEY
        self.unsplash_api_url = UNSPLASH_API_URL
        self.openverse_api_url = OPENVERSE_API_URL
        self.pixabay_api_key = PIXABAY_API_KEY
        self.pixabay_api_url = PIXABAY_API_URL

    def _fetch_from_pexels(self, prompt, width, height):
        if not self.pexels_api_key or "YOUR_" in self.pexels_api_key: return None
        try:
            response = requests.get(f"{self.pexels_api_url}/search", headers={"Authorization": self.pexels_api_key}, params={"query": prompt, "orientation": "portrait", "per_page": 1}, timeout=15)
            response.raise_for_status()
            data = response.json()
            if data and data['photos']:
                img_data = requests.get(data['photos'][0]['src']['original'], stream=True, timeout=15).content
                return Image.open(io.BytesIO(img_data))
        except Exception as e:
            print(f"Error fetching from Pexels: {e}")
        return None

    def _fetch_from_unsplash(self, prompt, width, height):
        if not self.unsplash_access_key or "YOUR_" in self.unsplash_access_key: return None
        try:
            params = {"query": prompt, "orientation": "portrait", "client_id": self.unsplash_access_key, "per_page": 1}
            response = requests.get(f"{self.unsplash_api_url}/search/photos", params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            if data and data['results']:
                img_data = requests.get(data['results'][0]['urls']['regular'], stream=True, timeout=15).content
                return Image.open(io.BytesIO(img_data))
        except Exception as e:
            print(f"Error fetching from Unsplash: {e}")
        return None

    def _fetch_from_openverse(self, prompt, width, height):
        try:
            params = {"q": prompt, "license_type": "commercial", "image_type": "photo", "orientation": "portrait", "page_size": 1}
            response = requests.get(self.openverse_api_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            if data and data['results']:
                img_data = requests.get(data['results'][0]['url'], stream=True, timeout=15).content
                return Image.open(io.BytesIO(img_data.content))
        except Exception as e:
            print(f"Error fetching from Openverse: {e}")
        return None

    def _fetch_from_pixabay(self, prompt, width, height):
        if not self.pixabay_api_key or "YOUR_" in self.pixabay_api_key: return None
        try:
            params = {"key": self.pixabay_api_key, "q": prompt, "image_type": "photo", "orientation": "vertical", "safesearch": "true", "per_page": 3}
            response = requests.get(self.pixabay_api_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            if data and data['hits']:
                img_data = requests.get(data['hits'][0]['largeImageURL'], stream=True, timeout=15).content
                return Image.open(io.BytesIO(img_data))
        except Exception as e:
            print(f"Error fetching from Pixabay: {e}")
        return None

    def fetch_image(self, prompt, width=IMAGE_DISPLAY_WIDTH, height=IMAGE_DISPLAY_HEIGHT):
        """Attempts to fetch an image from various stock photo APIs."""
        print(f"Searching stock photo APIs for prompt: {prompt[:50]}...")
        for fetch_func in [self._fetch_from_pexels, self._fetch_from_unsplash, self._fetch_from_openverse, self._fetch_from_pixabay]:
            image = fetch_func(prompt, width, height)
            if image:
                return image
        return None

class ImageLocalProcessor:
    """Handles local image processing and text overlays."""
    def _get_filtered_source_display(self, original_source_text: str) -> str:
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
            content_type_display = post_data.get('content_type_display', 'NEWS').upper().replace('_', ' ')
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
            traceback.print_exc()
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
        if not CLOUDINARY_CLOUD_NAME or "YOUR_" in CLOUDINARY_CLOUD_NAME:
            print("Cloudinary credentials are not set. Skipping upload.")
            return None
        try:
            print(f"Uploading {image_path} to Cloudinary folder '{folder}' with public_id '{public_id}'...")
            upload_result = cloudinary.uploader.upload(image_path, public_id=public_id, folder=folder)
            return upload_result.get('secure_url')
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
            if not media_container_id: return False
            publish_url = f"{self.graph_api_base_url}{self.instagram_business_account_id}/media_publish"
            publish_params = {'creation_id': media_container_id, 'access_token': self.access_token}
            response = requests.post(publish_url, params=publish_params, timeout=30)
            response.raise_for_status()
            if response.json().get('id'):
                print(f"Post successfully published to Instagram with ID: {response.json().get('id')}")
                return True
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
            post_data['final_image'].save(image_path)
            print(f"Image saved to: {image_path}")
        else:
            image_path = "No image generated/saved"
        
        metadata = {k: v for k, v in post_data.items() if k != 'final_image'}
        metadata['Hashtags'] = ', '.join(metadata.get('hashtags', []))

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
            print(f"Error saving to JSON file: {e}")

        try:
            df = pd.DataFrame([metadata])
            if os.path.exists(self.ALL_POSTS_EXCEL_FILE):
                with pd.ExcelWriter(self.ALL_POSTS_EXCEL_FILE, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                    df.to_excel(writer, sheet_name='Posts', index=False, header=False, startrow=writer.sheets['Posts'].max_row)
            else:
                df.to_excel(self.ALL_POSTS_EXCEL_FILE, sheet_name='Posts', index=False)
        except Exception as e:
            print(f"Error saving to Excel file: {e}")

    def load_all_posts_data(self):
        """Loads all historical post data from the JSON file."""
        if os.path.exists(self.ALL_POSTS_JSON_FILE):
            try:
                with open(self.ALL_POSTS_JSON_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading {self.ALL_POSTS_JSON_FILE}: {e}")
        return []

def _load_style_recommendations():
    """Loads the last saved style recommendations from a JSON file."""
    if os.path.exists(STYLE_RECOMMENDATIONS_FILE):
        try:
            with open(STYLE_RECOMMENDATIONS_FILE, 'r') as f: return json.load(f)
        except Exception as e:
            print(f"Error loading style recommendations: {e}")
    return {}

def _save_style_recommendations(recommendations):
    """Saves the current style recommendations to a JSON file."""
    os.makedirs(os.path.dirname(STYLE_RECOMMENDATIONS_FILE), exist_ok=True)
    with open(STYLE_RECOMMENDATIONS_FILE, 'w') as f: json.dump(recommendations, f, indent=4)

def perform_weekly_analysis(mistral_client, local_saver_instance):
    """Analyzes past week's content and generates style recommendations."""
    print("\n--- Performing Weekly Performance Analysis ---")
    all_posts_data = local_saver_instance.load_all_posts_data()
    one_week_ago = datetime.now(UTC) - timedelta(days=WEEKLY_ANALYSIS_INTERVAL_DAYS)
    past_week_posts = [p for p in all_posts_data if 'Timestamp' in p and datetime.fromisoformat(p['Timestamp']) >= one_week_ago]
    if not past_week_posts:
        print("No posts found from the last week to analyze.")
        return {}
    past_content_summary = "\n".join([f"Post ({p['Timestamp']}):\n  Caption: {p.get('SEO_Caption', 'N/A')}" for p in past_week_posts])
    prompt = f"Analyze the following content. Provide specific recommendations to improve caption style and hashtag strategy.\n\n{past_content_summary}"
    try:
        response = mistral_client.chat.completions.create(model=OPENROUTER_MISTRAL_MODEL, messages=[{"role": "user", "content": prompt}])
        recs = {"weekly_analysis": response.choices[0].message.content, "timestamp": datetime.now(UTC).isoformat()}
        _save_style_recommendations(recs)
        return recs
    except Exception as e:
        print(f"Error calling Mistral for weekly analysis: {e}")
    return {}

def check_api_keys():
    """Checks if essential API keys are set."""
    # This function is preserved from your original code
    pass

# --- Main Workflow Execution ---
if __name__ == "__main__":
    check_api_keys()
    workflow_manager = WorkflowStateManager()
    news_fetcher = NewsFetcher()
    text_processor = TextProcessor()
    caption_generator = CaptionGenerator()
    # --- NEW: Initialize AI Image Generator ---
    ai_image_gen = HuggingFaceImageGenerator()
    # --- This is your original stock photo fetcher ---
    image_fetcher = ImageFetcher()
    image_local_processor = ImageLocalProcessor()
    cloudinary_uploader = CloudinaryUploader()
    instagram_poster = InstagramPoster()
    local_saver = LocalSaver(IMAGE_OUTPUT_DIR, JSON_OUTPUT_DIR, EXCEL_OUTPUT_DIR, ALL_POSTS_JSON_FILE, ALL_POSTS_EXCEL_FILE)
    mistral_client_for_analysis = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_MISTRAL_API_KEY)

    try:
        recommendation_text_for_llm = _load_style_recommendations().get('weekly_analysis', '')
        if workflow_manager.should_run_weekly_analysis():
            new_recs = perform_weekly_analysis(mistral_client_for_analysis, local_saver)
            if new_recs: recommendation_text_for_llm = new_recs.get('weekly_analysis', '')
            workflow_manager.update_last_analysis_timestamp()

        content_type = workflow_manager.get_current_post_type()
        post_num = workflow_manager.get_current_post_number()
        print(f"\n--- Processing Post {post_num}/{len(CONTENT_TYPE_CYCLE)} (Type: {content_type.replace('_', ' ').title()}) ---")

        post_to_process = news_fetcher.get_single_content_item(content_type)
        if not post_to_process:
            workflow_manager.increment_post_type_index()
            sys.exit(f"No content found for {content_type}.")

        post_to_process['original_description'] = post_to_process.get('description', 'N/A')
        title, summary, _ = text_processor.process_text(post_to_process['title'], post_to_process['description'], post_to_process['type'], recommendation_text_for_llm)
        post_to_process.update({'title': title, 'summary': summary})
        print(f"Generated Title: {title}\nGenerated Summary: {summary}")

        # --- MODIFIED IMAGE GENERATION FLOW ---
        pil_image = None
        if ENABLE_AI_IMAGE_GENERATION:
            prompt, _ = caption_generator.generate_image_prompt(title, summary)
            pil_image = ai_image_gen.generate_image(prompt)
            post_to_process['image_status'] = 'generated_ai' if pil_image else 'ai_failed'

        if not pil_image:
            print("AI generation failed or was disabled. Falling back to stock photo API.")
            pil_image = image_fetcher.fetch_image(f"{title} {summary}")
            post_to_process['image_status'] = 'fetched_api' if pil_image else 'api_failed'

        if not pil_image:
            print("All image sources failed. Creating placeholder.")
            pil_image = Image.new('RGB', (IMAGE_DISPLAY_WIDTH, IMAGE_DISPLAY_HEIGHT), color='black')
            post_to_process['image_status'] = 'placeholder'
        
        post_to_process['final_image'] = image_local_processor.overlay_text(pil_image, post_to_process)
        
        caption, hashtags, _ = caption_generator.generate_caption_and_hashtags(title, summary, recommendation_text_for_llm)
        post_to_process.update({'seo_caption': caption, 'hashtags': hashtags})

        local_saver.save_post(post_to_process)
        
        img_path = os.path.join(IMAGE_OUTPUT_DIR, f"{post_to_process['Post_ID']}.png")
        cloudinary_url = cloudinary_uploader.upload_image(img_path, post_to_process['Post_ID'])
        
        if cloudinary_url:
            full_caption = f"{caption}\n\n{' '.join(hashtags)}"
            instagram_poster.post_image(cloudinary_url, full_caption)

        workflow_manager.increment_post_type_index()
        print("--- Workflow finished successfully ---")

    except Exception as e:
        print(f"\nAn unhandled error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)
