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
import sys # Import sys for exiting
from openai import OpenAI # Import OpenAI client for OpenRouter
import cloudinary # NEW: Import Cloudinary library
import cloudinary.uploader # NEW: Import Cloudinary uploader

# Fix for some SSL certificate issues with feedparser on some systems
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context


# Import configuration and state manager
from config import (
    # Removed NEWS_API_KEY, API_NINJAS_FACTS_API_KEY as per instructions
    OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_SITE_URL, OPENROUTER_SITE_NAME, # OpenRouter Deepseek config
    OPENROUTER_MISTRAL_API_KEY, OPENROUTER_MISTRAL_MODEL, # NEW: OpenRouter Mistral config
    PEXELS_API_KEY, PEXELS_API_URL, # Pexels config
    UNSPLASH_ACCESS_KEY, UNSPLASH_API_URL, # Unsplash config
    OPENVERSE_API_URL, # Openverse config
    PIXABAY_API_KEY, PIXABAY_API_URL, # Pixabay config
    CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET, # NEW: Cloudinary config
    FB_PAGE_ACCESS_TOKEN, INSTAGRAM_BUSINESS_ACCOUNT_ID, # NEW: Instagram Graph API config
    IMAGE_OUTPUT_DIR, JSON_OUTPUT_DIR, EXCEL_OUTPUT_DIR,
    ALL_POSTS_JSON_FILE, ALL_POSTS_EXCEL_FILE, STYLE_RECOMMENDATIONS_FILE, # NEW: STYLE_RECOMMENDATIONS_FILE
    WEEKLY_ANALYSIS_INTERVAL_DAYS, # NEW: WEEKLY_ANALYSIS_INTERVAL_DAYS
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
    CONTENT_TYPE_CYCLE # Added for workflow
)
from state_manager import WorkflowStateManager # This is already imported

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
    Uses a dummy ImageDraw.Draw object for accurate textbbox calculation.
    """
    if not text:
        return [""]

    words = text.split(' ')

    # Apply word count limit first
    original_text_length = len(words)
    if max_words is not None and original_text_length > max_words:
        words = words[:max_words]
        text_to_wrap = ' '.join(words) + "..." # Add ellipsis if truncated
    else:
        text_to_wrap = ' '.join(words) # Use full text if within limit or no limit

    lines = []
    current_line_words = []

    # Create a dummy draw object for accurate text width calculation
    dummy_img = Image.new('RGB', (1,1))
    dummy_draw = ImageDraw.Draw(dummy_img)

    for word in text_to_wrap.split(' '): # Split again in case ellipsis added
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
                    published_dt = datetime.now(UTC) # Fallback to now if no publish date

                if published_dt > time_threshold:
                    # Clean description: remove HTML tags and multiple spaces
                    raw_description = entry.summary if hasattr(entry, 'summary') and entry.summary else (entry.title if hasattr(entry, 'title') else 'No Description')
                    clean_description = re.sub(r'<[^>]+>', '', raw_description).strip() # Remove HTML tags
                    clean_description = re.sub(r'\s+', ' ', clean_description).strip() # Remove multiple spaces

                    recent_articles.append({
                        'title': entry.title if hasattr(entry, 'title') and entry.title else 'No Title',
                        'description': clean_description,
                        'url': entry.link if hasattr(entry, 'link') else rss_url,
                        'source': feed.feed.title if hasattr(feed.feed, 'title') and feed.feed.title else 'Unknown RSS', # RSS feed title is often the source
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
        Returns None if no recent, relevant content can be found.
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

        tech_news_sources = [ # NEW: Tech news RSS feeds
            {'type': 'rss', 'url': 'https://www.theverge.com/rss/index.xml', 'name': 'The Verge'},
            {'type': 'rss', 'url': 'https://techcrunch.com/feed/', 'name': 'TechCrunch'},
            {'type': 'rss', 'url': 'https://arstechnica.com/feed/', 'name': 'Ars Technica'},
            {'type': 'rss', 'url': 'https://www.wired.com/feed/rss', 'name': 'Wired'}
        ]

        environmental_news_sources = [ # NEW: Environmental news RSS feeds
            {'type': 'rss', 'url': 'https://www.sciencedaily.com/rss/earth_climate.xml', 'name': 'ScienceDaily Environment'},
            {'type': 'rss', 'url': 'https://www.mongabay.com/feed/', 'name': 'Mongabay'},
            {'type': 'rss', 'url': 'https://www.theguardian.com/environment/rss', 'name': 'The Guardian Environment'},
            {'type': 'rss', 'url': 'https://e360.yale.edu/digest.rss', 'name': 'Yale Environment 360'}
        ]


        content_item = None
        articles = []
        selected_sources = []

        if content_type == 'world_news':
            selected_sources = world_news_sources
        elif content_type == 'indian_news':
            selected_sources = indian_news_sources
        elif content_type == 'tech_news': # NEW
            selected_sources = tech_news_sources
        elif content_type == 'environmental_news': # NEW
            selected_sources = environmental_news_sources
        # Removed 'curiosity_fact' branch

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
            return None # Return None if no articles found from any source

        return content_item


class TextProcessor:
    """Summarizes and enhances text using OpenRouter AI (DeepSeek model)."""

    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.model = OPENROUTER_MODEL
        self.site_url = OPENROUTER_SITE_URL
        self.site_name = OPENROUTER_SITE_NAME
        self.client = OpenAI( # Initialize OpenAI client
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            # Removed 'headers' keyword argument as it's not supported here
        )
        self.dummy_draw = ImageDraw.Draw(Image.new('RGB', (1,1))) # For text bbox calculation

    def _call_ai_api(self, messages):
        """Helper to call the OpenRouter API using the OpenAI client. Returns (short_title, summary, success_flag)."""
        if not self.api_key:
            print("OPENROUTER_API_KEY is not set. Skipping AI text processing.")
            return "AI Key Error", "Please set your OpenRouter API key in config.py.", False

        try:
            completion = self.client.chat.completions.create(
                extra_headers={ # 'extra_headers' is the correct way to pass custom headers
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                },
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500,
                response_format={"type": "json_object"} # Ensure JSON output
            )

            if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
                ai_content_str = completion.choices[0].message.content
                try:
                    parsed_data = json.loads(ai_content_str)
                    short_title = parsed_data.get('short_title', "Untitled")
                    summary = parsed_data.get('summary_text', "No summary available.")

                    # Apply word limits as a safeguard, even if validation is removed
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

        except Exception as e: # Catch all exceptions from OpenAI client (e.g., APIError)
            print(f"Error calling OpenRouter API: {e}")
            return "API Error", f"AI API request failed: {e}", False


    def process_text(self, title, description, post_type, style_recommendations=""): # NEW: Added style_recommendations
        """Generates concise title and summary for a given post using OpenRouter AI (Deepseek).
        Returns (short_title, summary, success_flag).
        """
        messages = [
            {
                "role": "system",
                "content": f"""
                You are an expert content summarizer for social media news posts.
                Your task is to generate a very concise headline and a clear, standalone summary from provided news content.
                The summary MUST be comprehensive enough to explain the main incident or topic clearly.
                Adhere strictly to the word and line count constraints for the output format.
                The output MUST always be a valid JSON object.
                Consider the following style recommendations when generating the content: {style_recommendations}
                """
            },
            {
                "role": "user",
                "content": f"""
                Generate a summary and title for the following content.
                1. The title MUST be exactly {TITLE_MAX_WORDS} words long.
                2. The summary MUST be between {SUMMARY_MIN_WORDS} and {SUMMARY_MAX_WORDS} words.
                3. The summary should be approximately 4-6 lines long when formatted for display on a social media image (assuming a width of {CANVAS_WIDTH - (LEFT_PADDING + RIGHT_PADDING)} pixels with font size {FONT_SIZE_SUMMARY}, which means roughly 25-35 characters per line).
                4. The summary MUST be a complete thought and end gracefully, not abruptly.

                Content Type: {post_type.replace('_', ' ').title()}
                Original Title: {title}
                Original Description: {description}

                Return ONLY the JSON object with two keys: "short_title" and "summary_text".
                """
            }
        ]

        short_title, summary, success = self._call_ai_api(messages)

        if not success: # If AI call itself failed
            print("AI text processing failed. Using truncated original description as fallback for summary.")
            # Fallback for summary
            fallback_summary = ' '.join(description.split()[:SUMMARY_MAX_WORDS])
            font_summary = load_font(FONT_PATH_REGULAR, FONT_SIZE_SUMMARY)
            wrapped_fallback = wrap_text_by_word_count(fallback_summary, font_summary, CANVAS_WIDTH - (LEFT_PADDING + RIGHT_PADDING), max_words=SUMMARY_MAX_WORDS)
            final_summary = ' '.join(wrapped_fallback)

            # Fallback for title
            final_short_title = ' '.join(title.split()[:TITLE_MAX_WORDS])
            return final_short_title, final_summary, False # Indicate failure

        return short_title, summary, True


class CaptionGenerator:
    """Generates Instagram captions and hashtags using OpenRouter AI (Mistral model)."""

    def __init__(self):
        self.api_key = OPENROUTER_MISTRAL_API_KEY
        self.model = OPENROUTER_MISTRAL_MODEL
        self.site_url = OPENROUTER_SITE_URL # Using the same site URL as Deepseek
        self.site_name = OPENROUTER_SITE_NAME # Using the same site name as Deepseek
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            # Removed 'headers' keyword argument as it's not supported here
        )

    def generate_caption_and_hashtags(self, short_title, summary, style_recommendations=""): # NEW: Added style_recommendations
        """
        Generates an Instagram-style caption and 10 relevant hashtags.
        Returns (caption, hashtags_list, success_flag).
        """
        if not self.api_key:
            print("OPENROUTER_MISTRAL_API_KEY is not set. Skipping caption/hashtag generation.")
            return "Generated caption fallback.", ["#news", "#update"], False

        messages = [
            {
                "role": "system",
                "content": f"""
                You are a creative social media manager specializing in Instagram posts for news.
                Your task is to generate a concise and engaging Instagram caption and exactly 10 trending hashtags based on a news title and summary.
                The caption should be evocative and encourage engagement.
                The hashtags should be highly relevant to the topic and popular.
                The output MUST be a valid JSON object with keys "caption" (string) and "hashtags" (array of strings).
                Consider the following style recommendations when generating the content: {style_recommendations}
                """
            },
            {
                "role": "user",
                "content": f"""
                Generate an Instagram caption and 10 relevant hashtags.

                News Title: {short_title}
                News Summary: {summary}

                Return ONLY the JSON object with two keys: "caption" and "hashtags".
                Example: {{"caption": "Example caption...", "hashtags": ["#example", "#trending"]}}
                """
            }
        ]

        try:
            completion = self.client.chat.completions.create(
                extra_headers={ # 'extra_headers' is the correct way to pass custom headers
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                },
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=200, # Sufficient for caption and hashtags
                response_format={"type": "json_object"}
            )

            if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
                ai_content_str = completion.choices[0].message.content
                try:
                    parsed_data = json.loads(ai_content_str)
                    caption = parsed_data.get('caption', "Engaging caption from Mistral.")
                    hashtags = parsed_data.get('hashtags', [])
                    # Ensure hashtags is a list of strings
                    if not isinstance(hashtags, list) or not all(isinstance(h, str) for h in hashtags):
                        hashtags = ["#news", "#update"] # Fallback if format is wrong

                    # Ensure exactly 10 hashtags if possible, or truncate/pad
                    if len(hashtags) > 10:
                        hashtags = hashtags[:10]
                    elif len(hashtags) < 10:
                        # Add generic hashtags if not enough generated
                        generic_hashtags = ["#dailynews", "#breaking", "#insightpulse", "#info", "#currentaffairs"]
                        for gen_tag in generic_hashtags:
                            if len(hashtags) >= 10:
                                break
                            if gen_tag not in hashtags:
                                hashtags.append(gen_tag)

                    # Ensure hashtags start with #
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


class ImageFetcher:
    """Fetches images from Pexels, Unsplash, Openverse, and Pixabay based on text prompts."""

    def __init__(self):
        self.pexels_api_key = PEXELS_API_KEY
        self.pexels_api_url = PEXELS_API_URL
        self.unsplash_access_key = UNSPLASH_ACCESS_KEY
        self.unsplash_api_url = UNSPLASH_API_URL
        self.openverse_api_url = OPENVERSE_API_URL
        self.pixabay_api_key = PIXABAY_API_KEY
        self.pixabay_api_url = PIXABAY_API_URL

    def _fetch_from_pexels(self, prompt, width, height):
        """Attempts to fetch an image from Pexels."""
        try:
            if not self.pexels_api_key:
                print("PEXELS_API_KEY is not set. Skipping Pexels.")
                return None

            headers = {
                "Authorization": self.pexels_api_key
            }
            params = {
                "query": prompt,
                "orientation": "portrait",
                "size": "large",
                "per_page": 1
            }
            print(f"Searching Pexels for image with prompt: {prompt[:50]}...")
            response = requests.get(f"{self.pexels_api_url}/search", headers=headers, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            if data and data['photos']:
                image_url = data['photos'][0]['src']['original']
                print(f"Found image on Pexels: {image_url}")
                img_data = requests.get(image_url, stream=True, timeout=15)
                img_data.raise_for_status()
                return Image.open(io.BytesIO(img_data.content))
            return None
        except requests.exceptions.Timeout:
            print(f"Pexels API request timed out for prompt '{prompt[:50]}...'.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching from Pexels: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during Pexels fetch: {e}")
            return None

    def _fetch_from_unsplash(self, prompt, width, height):
        """Attempts to fetch an image from Unsplash."""
        try:
            if not self.unsplash_access_key:
                print("UNSPLASH_ACCESS_KEY is not set. Skipping Unsplash.")
                return None

            params = {
                "query": prompt,
                "orientation": "portrait",
                "client_id": self.unsplash_access_key,
                "per_page": 1
            }
            print(f"Searching Unsplash for image with prompt: {prompt[:50]}...")
            response = requests.get(f"{self.unsplash_api_url}/search/photos", params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            if data and data['results']:
                image_url = data['results'][0]['urls']['regular']
                print(f"Found image on Unsplash: {image_url}")
                img_data = requests.get(image_url, stream=True, timeout=15)
                img_data.raise_for_status()
                return Image.open(io.BytesIO(img_data.content))
            return None
        except requests.exceptions.Timeout:
            print(f"Unsplash API request timed out for prompt '{prompt[:50]}...'.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching from Unsplash: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during Unsplash fetch: {e}")
            return None

    def _fetch_from_openverse(self, prompt, width, height):
        """Attempts to fetch an image from Openverse."""
        try:
            params = {
                "q": prompt,
                "license_type": "commercial",
                "image_type": "photo",
                "orientation": "portrait",
                "page_size": 1
            }
            print(f"Searching Openverse for image with prompt: {prompt[:50]}...")
            response = requests.get(self.openverse_api_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            if data and data['results']:
                image_url = data['results'][0]['url']
                print(f"Found image on Openverse: {image_url}")
                img_data = requests.get(image_url, stream=True, timeout=15)
                img_data.raise_for_status()
                return Image.open(io.BytesIO(img_data.content))
            return None
        except requests.exceptions.Timeout:
            print(f"Openverse API request timed out for prompt '{prompt[:50]}...'.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching from Openverse: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during Openverse fetch: {e}")
            return None

    def _fetch_from_pixabay(self, prompt, width, height):
        """Attempts to fetch an image from Pixabay."""
        try:
            if not self.pixabay_api_key:
                print("PIXABAY_API_KEY is not set. Skipping Pixabay.")
                return None

            params = {
                "key": self.pixabay_api_key,
                "q": prompt,
                "image_type": "photo",
                "orientation": "vertical",
                "safesearch": "true",
                "per_page": 1,
                "editors_choice": "true",
                "min_width": width,
                "min_height": height
            }
            print(f"Searching Pixabay for image with prompt: {prompt[:50]}...")
            response = requests.get(self.pixabay_api_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            if data and data['hits']:
                image_url = data['hits'][0].get('largeImageURL') or data['hits'][0].get('webformatURL')
                if image_url:
                    print(f"Found image on Pixabay: {image_url}")
                    img_data = requests.get(image_url, stream=True, timeout=15)
                    img_data.raise_for_status()
                    return Image.open(io.BytesIO(img_data.content))
            return None
        except requests.exceptions.Timeout:
            print(f"Pixabay API request timed out for prompt '{prompt[:50]}...'.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching from Pixabay: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during Pixabay fetch: {e}")
            return None


    def fetch_image(self, prompt, width=IMAGE_DISPLAY_WIDTH, height=IMAGE_DISPLAY_HEIGHT):
        """
        Attempts to fetch an image, prioritizing Pexels, then Unsplash, Openverse, then Pixabay.
        Returns a PIL Image object or None if no image could be fetched.
        """
        fetched_image = None

        # Try Pexels first
        fetched_image = self._fetch_from_pexels(prompt, width, height)

        # If Pexels fails, try Unsplash
        if fetched_image is None:
            fetched_image = self._fetch_from_unsplash(prompt, width, height)

        # If Unsplash fails, try Openverse
        if fetched_image is None:
            fetched_image = self._fetch_from_openverse(prompt, width, height)

        # If Openverse fails, try Pixabay
        if fetched_image is None:
            fetched_image = self._fetch_from_pixabay(prompt, width, height)

        return fetched_image # Return None if all failed, placeholder will be made later


class ImageLocalProcessor:
    """Handles local image processing and text overlays."""

    def _get_filtered_source_display(self, original_source_text: str) -> str:
        """
        Filters and shortens the source text for display in the 'Source - X' box.
        Prioritizes known concise names and ensures length <= 3 words.
        """
        if not original_source_text:
            return "UNKNOWN"

        # Explicit mappings for common sources to desired short forms
        source_map = {
            "SYSTEM": "SYSTEM",
            "API-NINJAS FACTS": "API-NINJAS",
            "BBC WORLD NEWS": "BBC",
            "BBC INDIA": "BBC",
            "REUTERS WORLD NEWS": "REUTERS",
            "NYT WORLD NEWS": "NYT",
            "AL JAZEERA": "AL JAZEERA",
            "THE HINDU": "HINDU",
            "TIMES OF INDIA": "TIMES INDIA",
            "ZEE NEWS INDIA": "ZEE NEWS",
            "NEWSAPI": "NEWSAPI",
            "FINANCIAL POST": "FINANCIAL POST",
            "THE VERGE": "THE VERGE", # NEW
            "TECHCRUNCH": "TECHCRUNCH", # NEW
            "ARS TECHNICA": "ARS TECHNICA", # NEW
            "WIRED": "WIRED", # NEW
            "SCIENCEDAILY ENVIRONMENT": "SCIENCEDAILY", # NEW
            "MONGABAY": "MONGABAY", # NEW
            "THE GUARDIAN ENVIRONMENT": "GUARDIAN ENV", # NEW
            "YALE ENVIRONMENT 360": "YALE E360" # NEW
        }

        # Check if an explicit mapping exists
        mapped_source = source_map.get(original_source_text.upper().strip())
        if mapped_source:
            return mapped_source

        # If no explicit mapping, apply generic filtering
        processed_source = original_source_text.upper().strip()

        noisy_phrases = [
            r'\bNEWS\b', r'\bREPORTS\b', r'\bLIVE\b', r'\bUPDATE\b', r'\bVIDEO FROM\b',
            r'\bBREAKING\b', r'\bGLOBAL\b', r'\bWORLD\b', r'\bINDIA\b', r'\bTHE\b',
            r'\bCOM\b', r'\.COM', r'\.ORG', r'\.NET', r'\.IN', r'\.CO\.IN',
            r'INTERNATIONAL EDITION', r'LATEST TODAY', r'CORRESPONDENT', r'CHANNEL', r'TV', r'PRESS'
        ]

        for phrase in noisy_phrases:
            processed_source = re.sub(phrase, '', processed_source).strip()

        processed_source = re.sub(r'\s+', ' ', processed_source).strip()

        words = processed_source.split()
        if len(words) <= 3 and len(words) > 0:
            return processed_source
        elif words:
            return ' '.join(words[:3])

        return "UNKNOWN"


    def overlay_text(self, base_pil_image, post_data):
        """
        Creates the final post image with gradient background, central image, and text overlays.
        This function dynamically adjusts element positions to prevent overlap.
        """
        try:
            # 1. Create Gradient Background
            background_gen = BackgroundGenerator()
            final_canvas = background_gen.generate_gradient_background(CANVAS_WIDTH, CANVAS_HEIGHT,
                                                                        COLOR_GRADIENT_TOP_LEFT, COLOR_GRADIENT_BOTTOM_RIGHT)
            draw = ImageDraw.Draw(final_canvas)
            dummy_draw_for_text_bbox = ImageDraw.Draw(Image.new('RGB', (1,1)))


            # --- TOP LEFT CATEGORY TEXT (e.g., "WORLD NEWS") ---
            content_type_display = post_data.get('content_type_display', 'NEWS').upper()

            if 'WORLD_NEWS' in content_type_display:
                top_left_text = "WORLD NEWS"
            elif 'INDIAN_NEWS' in content_type_display:
                top_left_text = "INDIAN NEWS"
            elif 'TECH_NEWS' in content_type_display: # NEW
                top_left_text = "TECH NEWS"
            elif 'ENVIRONMENTAL_NEWS' in content_type_display: # NEW
                top_left_text = "ENVIRONMENT NEWS"
            else:
                top_left_text = "GENERAL NEWS" # Fallback


            font_top_left_text = load_font(FONT_PATH_BOLD, FONT_SIZE_TOP_LEFT_TEXT)
            draw.text((TOP_LEFT_TEXT_POS_X, TOP_LEFT_TEXT_POS_Y), top_left_text, font=font_top_left_text, fill=COLOR_LIGHT_GRAY_TEXT)


            # --- TIMESTAMP (TOP RIGHT) ---
            timestamp_text = datetime.now().strftime("%d/%m/%Y %H:%M")
            font_timestamp = load_font(FONT_PATH_BOLD, FONT_SIZE_TIMESTAMP)
            timestamp_bbox = dummy_draw_for_text_bbox.textbbox((0,0), timestamp_text, font=font_timestamp)
            timestamp_width = timestamp_bbox[2] - timestamp_bbox[0]

            draw.text((TIMESTAMP_POS_X_RIGHT_ALIGN - timestamp_width, TIMESTAMP_POS_Y),
                      timestamp_text, font=font_timestamp, fill=COLOR_WHITE)

            # --- NEWS IMAGE (CENTERED, ROUNDED CORNERS) ---
            image_start_y = max(
                TOP_LEFT_TEXT_POS_Y + (dummy_draw_for_text_bbox.textbbox((0,0), top_left_text, font=font_top_left_text)[3] - dummy_draw_for_text_bbox.textbbox((0,0), top_left_text, font=font_top_left_text)[1]),
                TIMESTAMP_POS_Y + (timestamp_bbox[3] - timestamp_bbox[1])
            ) + IMAGE_TOP_MARGIN_FROM_TOP_ELEMENTS

            # Resize/Crop the base_pil_image to fit IMAGE_DISPLAY_WIDTH x IMAGE_DISPLAY_HEIGHT
            target_aspect_ratio = IMAGE_DISPLAY_WIDTH / IMAGE_DISPLAY_HEIGHT
            generated_aspect_ratio = base_pil_image.width / base_pil_image.height

            if generated_aspect_ratio > target_aspect_ratio:
                new_width = int(base_pil_image.height * target_aspect_ratio)
                left = (base_pil_image.width - new_width) / 2
                top = 0
                right = (base_pil_image.width + new_width) / 2
                bottom = base_pil_image.height
            else:
                new_height = int(base_pil_image.width / target_aspect_ratio)
                left = 0
                top = (base_pil_image.height - new_height) / 2
                right = base_pil_image.width
                bottom = (base_pil_image.height + new_height) / 2

            cropped_image = base_pil_image.crop((left, top, right, bottom))
            news_image_for_display = cropped_image.resize((IMAGE_DISPLAY_WIDTH, IMAGE_DISPLAY_HEIGHT), Image.Resampling.LANCZOS)

            # Create a mask for rounded corners
            mask = Image.new('L', (IMAGE_DISPLAY_WIDTH, IMAGE_DISPLAY_HEIGHT), 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.rounded_rectangle((0, 0, IMAGE_DISPLAY_WIDTH, IMAGE_DISPLAY_HEIGHT),
                                        radius=IMAGE_ROUND_RADIUS, fill=255)

            news_image_x = (CANVAS_WIDTH - IMAGE_DISPLAY_WIDTH) // 2

            final_canvas.paste(news_image_for_display, (news_image_x, int(image_start_y)), mask)


            # --- TITLE (BELOW IMAGE) ---
            title_text_raw = str(post_data.get('title', 'NO TITLE')).upper()
            font_headline = load_font(FONT_PATH_EXTRABOLD, FONT_SIZE_HEADLINE)

            wrapped_title_lines = wrap_text_by_word_count(title_text_raw, font_headline,
                                                        CANVAS_WIDTH - (LEFT_PADDING + RIGHT_PADDING),
                                                        max_words=TITLE_MAX_WORDS)

            current_y_title = image_start_y + IMAGE_DISPLAY_HEIGHT + TITLE_TOP_MARGIN_FROM_IMAGE

            for line in wrapped_title_lines:
                line_bbox = dummy_draw_for_text_bbox.textbbox((0,0), line, font=font_headline)
                line_width = line_bbox[2] - line_bbox[0]

                text_x_centered = (CANVAS_WIDTH - line_width) / 2
                draw.text((text_x_centered, current_y_title), line, font=font_headline, fill=COLOR_WHITE)
                current_y_title += (line_bbox[3] - line_bbox[1]) + TITLE_LINE_SPACING

            # --- SUMMARY TEXT (BELOW TITLE) - Dynamically positioned ---
            summary_text_raw = str(post_data.get('summary', 'No summary provided.')).replace("&#x27;", "'").replace("&quot;", "\"")
            font_summary = load_font(FONT_PATH_REGULAR, FONT_SIZE_SUMMARY)

            wrapped_summary_lines = wrap_text_by_word_count(summary_text_raw, font_summary,
                                                           CANVAS_WIDTH - (LEFT_PADDING + RIGHT_PADDING),
                                                           max_words=SUMMARY_MAX_WORDS)

            current_y_summary = current_y_title + SUMMARY_TOP_MARGIN_FROM_TITLE

            for line in wrapped_summary_lines:
                draw.text((LEFT_PADDING, current_y_summary), line, font=font_summary, fill=COLOR_WHITE)
                line_height = (dummy_draw_for_text_bbox.textbbox((0,0), line, font=font_summary)[3] - dummy_draw_for_text_bbox.textbbox((0,0), line, font=font_summary)[1])
                current_y_summary += line_height + SUMMARY_LINE_SPACING

            current_y_summary -= SUMMARY_LINE_SPACING


            # --- DIVIDER LINE (THIN, FAINT RED) - Dynamically positioned ---
            divider_y = current_y_summary + DIVIDER_Y_OFFSET_FROM_SUMMARY
            color_red_alpha = (COLOR_RED[0], COLOR_RED[1], COLOR_RED[2], int(255 * 0.3))
            draw.line([(LEFT_PADDING, divider_y), (CANVAS_WIDTH - RIGHT_PADDING, divider_y)], fill=color_red_alpha, width=DIVIDER_LINE_THICKNESS)


            # --- Calculate Y-positions for bottom block elements (aligned on shared baseline) ---
            # Calculate height of the source box first, as it will anchor the bottom layout
            filtered_source_for_display = self._get_filtered_source_display(post_data.get('source', 'SOURCE'))
            source_text_display = f"Source - {filtered_source_for_display}"
            font_source = load_font(FONT_PATH_REGULAR, FONT_SIZE_SOURCE)
            source_bbox_content = dummy_draw_for_text_bbox.textbbox((0,0), source_text_display, font=font_source)
            source_width_content = source_bbox_content[2] - source_bbox_content[0]
            source_height_content = source_bbox_content[3] - source_bbox_content[1]

            source_rect_width = source_width_content + (2 * SOURCE_RECT_PADDING_X)
            source_rect_height = source_height_content + (2 * SOURCE_RECT_PADDING_Y)

            # Bottom edge of the source rectangle aligns with CANVAS_HEIGHT - BOTTOM_PADDING
            source_rect_y2 = CANVAS_HEIGHT - BOTTOM_PADDING
            source_rect_y1 = source_rect_y2 - source_rect_height
            source_text_y = source_rect_y1 + SOURCE_RECT_PADDING_Y

            # For logo, align its vertical center with the source box's vertical center
            logo_y = source_rect_y1 + (source_rect_height - LOGO_HEIGHT) / 2


            # --- LOGO (BOTTOM LEFT) ---
            try:
                # Use the path directly from config.py which is now correct
                logo_image = Image.open(LOGO_PATH).convert("RGBA")
                # Resize while maintaining aspect ratio
                logo_image.thumbnail((LOGO_WIDTH, LOGO_HEIGHT), Image.Resampling.LANCZOS)

                # Paste the logo. If logo has transparency, use it as the mask.
                final_canvas.paste(logo_image, (LEFT_PADDING, int(logo_y)), logo_image)

            except FileNotFoundError:
                print(f"Warning: Logo file not found at {LOGO_PATH}. Embedding text fallback.")
                draw.text((LEFT_PADDING, int(logo_y) + (LOGO_HEIGHT - 30) // 2), "Insight Pulse", font=load_font(FONT_PATH_BOLD, 30), fill=COLOR_WHITE)
            except Exception as e:
                print(f"Error embedding logo: {e}. Embedding text fallback.")
                draw.text((LEFT_PADDING, int(logo_y) + (LOGO_HEIGHT - 30) // 2), "Insight Pulse (Error)", font=load_font(FONT_PATH_BOLD, 30), fill=COLOR_WHITE)


            # --- SOURCE (BOTTOM RIGHT) ---
            source_rect_x1 = SOURCE_POS_X_RIGHT_ALIGN - source_rect_width
            source_rect_x2 = SOURCE_POS_X_RIGHT_ALIGN

            draw.rounded_rectangle((source_rect_x1, source_rect_y1, source_rect_x2, source_rect_y2),
                                  radius=SOURCE_RECT_RADIUS, fill=COLOR_DARK_GRAY)
            draw.text((source_rect_x1 + SOURCE_RECT_PADDING_X, int(source_text_y)),
                      source_text_display, font=font_source, fill=COLOR_WHITE)


            return final_canvas

        except Exception as e:
            print(f"Error during image overlay and composition: {e}")
            import traceback
            traceback.print_exc()
            img_error = Image.new('RGB', (CANVAS_WIDTH, CANVAS_HEIGHT), color = (50, 50, 50))
            draw_error = ImageDraw.Draw(img_error)
            error_font = load_font(FONT_PATH_REGULAR, 40)
            draw_error.text((50, CANVAS_HEIGHT // 2 - 20), "IMAGE COMPOSITION ERROR", font=error_font, fill=(255,0,0))
            draw_error.text((50, CANVAS_HEIGHT // 2 + 30), f"Check API keys or prompt: {str(e)[:100]}...", font=load_font(FONT_PATH_REGULAR, 20), fill=(255,255,255))
            return img_error


class CloudinaryUploader:
    """Handles uploading images to Cloudinary."""

    def __init__(self):
        cloudinary.config(
            cloud_name=CLOUDINARY_CLOUD_NAME,
            api_key=CLOUDINARY_API_KEY,
            api_secret=CLOUDINARY_API_SECRET
        )

    def upload_image(self, image_path, public_id, folder="news_posts"):
        """
        Uploads an image to Cloudinary.
        image_path: local path to the image file.
        public_id: A unique identifier for the image in Cloudinary.
        folder: The folder in Cloudinary to upload to.
        Returns the secure URL of the uploaded image or None on failure.
        """
        try:
            if not CLOUDINARY_CLOUD_NAME or not CLOUDINARY_API_KEY or not CLOUDINARY_API_SECRET:
                print("Cloudinary credentials are not set. Skipping upload.")
                return None

            print(f"Uploading {image_path} to Cloudinary folder '{folder}' with public_id '{public_id}'...")
            upload_result = cloudinary.uploader.upload(
                image_path,
                public_id=public_id,
                folder=folder
            )
            secure_url = upload_result.get('secure_url')
            if secure_url:
                print(f"Image uploaded to Cloudinary: {secure_url}")
                return secure_url
            else:
                print(f"Cloudinary upload failed: No secure_url in response. Result: {upload_result}")
                return None
        except Exception as e:
            print(f"Error uploading image to Cloudinary: {e}")
            return None


class InstagramPoster:
    """Handles posting images to Instagram via the Facebook Graph API."""

    def __init__(self):
        self.access_token = FB_PAGE_ACCESS_TOKEN
        self.instagram_business_account_id = INSTAGRAM_BUSINESS_ACCOUNT_ID
        self.graph_api_base_url = "https://graph.facebook.com/v19.0/" # Using a recent API version

    def post_image(self, image_url, caption):
        """
        Posts an image to Instagram.
        image_url: The secure URL of the image from Cloudinary.
        caption: The combined caption and hashtags.
        Returns True on success, False on failure.
        """
        if not self.access_token or not self.instagram_business_account_id or self.instagram_business_account_id == "YOUR_INSTAGRAM_BUSINESS_ACCOUNT_ID":
            print("Instagram Graph API credentials are not fully set or placeholder ID is used. Skipping Instagram post.")
            return False

        if not image_url:
            print("No image URL provided for Instagram post. Skipping.")
            return False

        print(f"Attempting to post to Instagram (Account ID: {self.instagram_business_account_id})...")

        # Step 1: Create media container
        media_container_url = f"{self.graph_api_base_url}{self.instagram_business_account_id}/media"
        media_params = {
            'image_url': image_url,
            'caption': caption,
            'access_token': self.access_token
        }
        try:
            response = requests.post(media_container_url, params=media_params, timeout=30)
            response.raise_for_status() # Raise an exception for HTTP errors
            media_container_id = response.json().get('id')
            print(f"Media container created with ID: {media_container_id}")
        except requests.exceptions.Timeout:
            print(f"Error: Instagram API (media container) request timed out. Image URL: {image_url}")
            return False
        except requests.exceptions.RequestException as e:
            print(f"Error creating Instagram media container: {e}. Response: {response.json() if 'response' in locals() else 'N/A'}")
            return False
        except Exception as e:
            print(f"An unexpected error occurred during Instagram media container creation: {e}")
            return False

        if not media_container_id:
            print("Failed to get media container ID.")
            return False

        # Step 2: Publish media container
        publish_url = f"{self.graph_api_base_url}{self.instagram_business_account_id}/media_publish"
        publish_params = {
            'creation_id': media_container_id,
            'access_token': self.access_token
        }
        try:
            response = requests.post(publish_url, params=publish_params, timeout=30)
            response.raise_for_status()
            post_id = response.json().get('id')
            if post_id:
                print(f"Post successfully published to Instagram with ID: {post_id}")
                return True
            else:
                print(f"Instagram publish failed: No post ID in response. Result: {response.json()}")
                return False
        except requests.exceptions.Timeout:
            print(f"Error: Instagram API (media publish) request timed out for container ID: {media_container_id}")
            return False
        except requests.exceptions.RequestException as e:
            print(f"Error publishing to Instagram: {e}. Response: {response.json() if 'response' in locals() else 'N/A'}")
            return False
        except Exception as e:
            print(f"An unexpected error occurred during Instagram publish: {e}")
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
        # Use a consistent timestamp for saving locally to match for Cloudinary public_id
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        post_id = f"{post_type_label}_{timestamp_str}_post-{workflow_manager.get_current_post_number()}"

        # --- FIX: Explicitly add Post_ID to the post_data dictionary ---
        post_data['Post_ID'] = post_id
        # --- END FIX ---

        # 1. Save Image
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
            print(f"Warning: No valid 'final_image' found in post_data for post ID {post_id}. Image not saved.")
            image_path = "No image generated/saved"

        # Prepare metadata for JSON/Excel
        # Now, 'Post_ID' is guaranteed to be in post_data
        metadata = {
            "Post_ID": post_data['Post_ID'], # Directly use from post_data
            "Title": post_data.get('title'),
            "Summary": post_data.get('summary'),
            "SEO_Caption": post_data.get('seo_caption'),
            "Hashtags": ', '.join(post_data.get('hashtags', [])), # Convert list to string for Excel
            "Local_Image_Path": image_path,
            "Cloudinary_URL": post_data.get('cloudinary_url', 'N/A'), # NEW
            "Instagram_Posted": post_data.get('instagram_posted', False), # NEW
            "Timestamp": datetime.now(UTC).isoformat(), # Ensure UTC for consistency
            "Source_Type": post_data.get('type'),
            "Source_URL": post_data.get('url', ''),
            "Original_Source": post_data.get('source', 'N/A'),
            "Original_Description": post_data.get('original_description', 'N/A')
        }

        # 2. Save to JSON
        try:
            existing_data = []
            if os.path.exists(self.ALL_POSTS_JSON_FILE):
                with open(self.ALL_POSTS_JSON_FILE, 'r', encoding='utf-8') as f:
                    try:
                        existing_data = json.load(f)
                        if not isinstance(existing_data, list):
                            existing_data = []
                    except json.JSONDecodeError:
                        print(f"Warning: {self.ALL_POSTS_JSON_FILE} is corrupted. Starting with empty JSON list.")
                        existing_data = []

            existing_data.append(metadata)

            with open(self.ALL_POSTS_JSON_FILE, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=4)
            print(f"Metadata appended to: {self.ALL_POSTS_JSON_FILE}")
        except Exception as e:
            print(f"Error saving to JSON file {self.ALL_POSTS_JSON_FILE}: {e}")

        # 3. Save to Excel
        try:
            df = pd.DataFrame([metadata])
            if os.path.exists(self.ALL_POSTS_EXCEL_FILE):
                with pd.ExcelWriter(self.ALL_POSTS_EXCEL_FILE, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                    # Check if the sheet has content beyond just headers
                    sheet_exists_and_has_data = False
                    if 'Posts' in writer.sheets:
                        if writer.sheets['Posts'].max_row > 1: # Row 1 is header
                            sheet_exists_and_has_data = True

                    if sheet_exists_and_has_data:
                        df.to_excel(writer, sheet_name='Posts', index=False, header=False, startrow=writer.sheets['Posts'].max_row)
                    else:
                        df.to_excel(writer, sheet_name='Posts', index=False, header=True)
            else:
                df.to_excel(self.ALL_POSTS_EXCEL_FILE, sheet_name='Posts', index=False)
            print(f"Metadata appended to: {self.ALL_POSTS_EXCEL_FILE}")
        except Exception as e:
            print(f"Error saving to Excel file {self.ALL_POSTS_EXCEL_FILE}: {e}")

    def load_all_posts_data(self): # NEW: Method to load all posts for analysis
        """Loads all historical post data from the JSON file."""
        if os.path.exists(self.ALL_POSTS_JSON_FILE):
            try:
                with open(self.ALL_POSTS_JSON_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    else:
                        print(f"Warning: {self.ALL_POSTS_JSON_FILE} content is not a list. Returning empty list.")
                        return []
            except json.JSONDecodeError:
                print(f"Warning: {self.ALL_POSTS_JSON_FILE} is corrupted. Returning empty list.")
                return []
            except Exception as e:
                print(f"Error loading all posts data from {self.ALL_POSTS_JSON_FILE}: {e}. Returning empty list.")
                return []
        print(f"No existing posts data file found at {self.ALL_POSTS_JSON_FILE}. Returning empty list.")
        return []


# --- NEW: Weekly Analysis Functions (moved from top level to module level or inside a class if makes sense) ---
# Keeping them as module-level functions for now as they interact with clients and other savers.

def _load_style_recommendations():
    """Loads the last saved style recommendations from a JSON file."""
    if os.path.exists(STYLE_RECOMMENDATIONS_FILE):
        try:
            with open(STYLE_RECOMMENDATIONS_FILE, 'r') as f:
                recommendations = json.load(f)
            print(f"Loaded style recommendations from {STYLE_RECOMMENDATIONS_FILE}")
            return recommendations
        except json.JSONDecodeError:
            print(f"Style recommendations file {STYLE_RECOMMENDATIONS_FILE} is corrupted. Returning empty recommendations.")
            return {}
        except Exception as e:
            print(f"Error loading style recommendations: {e}. Returning empty recommendations.")
            return {}
    print("No existing style recommendations file found. Returning empty recommendations.")
    return {}

def _save_style_recommendations(recommendations):
    """Saves the current style recommendations to a JSON file."""
    os.makedirs(os.path.dirname(STYLE_RECOMMENDATIONS_FILE), exist_ok=True)
    try:
        with open(STYLE_RECOMMENDATIONS_FILE, 'w') as f:
            json.dump(recommendations, f, indent=4)
        print(f"Saved style recommendations to {STYLE_RECOMMENDATIONS_FILE}")
    except Exception as e:
        print(f"Error saving style recommendations: {e}")

def perform_weekly_analysis(mistral_client, local_saver_instance): # Changed to accept local_saver instance
    """
    Analyzes past week's content using Mistral model and generates style recommendations.
    Args:
        mistral_client: The initialized OpenAI client for Mistral.
        local_saver_instance: An instance of LocalSaver to load historical data.
    Returns:
        A dictionary of new style recommendations.
    """
    print("\n--- Performing Weekly Performance Analysis ---")
    
    # Load all historical posts for analysis using the provided local_saver_instance
    all_posts_data = local_saver_instance.load_all_posts_data()

    # Filter posts from the last week
    one_week_ago = datetime.now(UTC) - timedelta(days=WEEKLY_ANALYSIS_INTERVAL_DAYS)
    
    past_week_posts = []
    for post in all_posts_data:
        try:
            # Check if 'Timestamp' exists and is a string before parsing
            if 'Timestamp' in post and isinstance(post['Timestamp'], str):
                post_timestamp = datetime.fromisoformat(post['Timestamp']).replace(tzinfo=UTC)
                if post_timestamp >= one_week_ago:
                    past_week_posts.append({
                        "timestamp": post['Timestamp'],
                        "content_type": post.get('Source_Type', 'N/A'), # Using 'Source_Type' from saved metadata
                        "seo_caption": post.get('SEO_Caption', 'N/A'),
                        "hashtags": post.get('Hashtags', 'N/A')
                    })
            else:
                print(f"Warning: Post data missing valid 'Timestamp' or is not string: {post}. Skipping post for analysis.")
                continue
        except KeyError as e:
            print(f"Warning: Post data missing key for analysis: {e}. Skipping post.")
            continue
        except ValueError as e:
            print(f"Warning: Could not parse timestamp for post: {e}. Skipping post.")
            continue

    if not past_week_posts:
        print("No posts found from the last week to analyze. Returning empty recommendations.")
        return {}

    # Prepare data for the prompt
    past_content_summary = ""
    for i, post in enumerate(past_week_posts):
        past_content_summary += (
            f"Post {i+1} ({post['timestamp']}):\n"
            f"  Type: {post['content_type']}\n"
            f"  Caption: {post['seo_caption']}\n"
            f"  Hashtags: {post['hashtags']}\n"
            "--------------------\n"
        )
    
    system_message = "You are an expert social media content strategist. Your task is to analyze past content and provide actionable, creative recommendations to improve future content's reach and engagement on Instagram. Think about caption style, hashtag strategy, and content themes/news focus. Be specific and provide examples."
    prompt = f"""Analyze the following past week's generated Instagram content. Based on this, provide specific and actionable recommendations to improve the 'style' of future posting content, news themes, captions, and hashtag usage for better reach.

Past Week's Content Data:
---
{past_content_summary}
---

Provide your recommendations in a structured text format, for example:

**Caption Style Recommendations:**
- Use more concise sentences.
- Incorporate a call-to-action (e.g., "What are your thoughts?").

**Hashtag Strategy Recommendations:**
- Mix broad and niche hashtags.
- Research trending hashtags related to environmental news.

**Content Theme/News Focus Recommendations:**
- Explore more positive or solution-oriented news stories.
- Increase coverage of tech innovations and breakthroughs.

Ensure your recommendations are clear and directly applicable to the content generation process.
"""

    print("Sending past content data to Mistral for analysis...")
    try:
        response = mistral_client.chat.completions.create( # Use the passed mistral_client
            model=OPENROUTER_MISTRAL_MODEL,
            messages=[{"role": "system", "content": system_message}, {"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7,
            extra_headers={"HTTP-Referer": OPENROUTER_SITE_URL, "X-Site-Name": OPENROUTER_SITE_NAME}, # Correctly pass headers
        )
        recommendations_text = response.choices[0].message.content.strip()

        if recommendations_text:
            print("Mistral Analysis Result:\n", recommendations_text)
            new_recommendations = {"weekly_analysis": recommendations_text, "timestamp": datetime.now(UTC).isoformat()}
            _save_style_recommendations(new_recommendations)
            return new_recommendations
        else:
            print("Failed to get style recommendations from Mistral (empty response).")
            return {}
    except Exception as e:
        print(f"Error calling Mistral for weekly analysis: {e}")
        return {}


def check_api_keys():
    """Checks if essential API keys are empty strings. Provides warnings."""
    warnings = []
    if not OPENROUTER_API_KEY:
        warnings.append("OPENROUTER_API_KEY (for Deepseek) is empty. AI text processing might be limited or fail.")
    if not PEXELS_API_KEY:
        warnings.append("PEXELS_API_KEY is empty. Pexels image fetches might be limited or fail.")
    if not UNSPLASH_ACCESS_KEY:
        warnings.append("UNSPLASH_ACCESS_KEY is empty. Unsplash image fetches might be limited or fail.")
    if not PIXABAY_API_KEY:
        warnings.append("PIXABAY_API_KEY is empty. Pixabay image fetches might be limited or fail.")
    if not OPENROUTER_MISTRAL_API_KEY:
        warnings.append("OPENROUTER_MISTRAL_API_KEY (for Mistral) is empty. Caption/hashtag generation and weekly analysis might fail.")
    if not CLOUDINARY_CLOUD_NAME or not CLOUDINARY_API_KEY or not CLOUDINARY_API_SECRET:
        warnings.append("Cloudinary API credentials are not fully set. Image upload will fail.")
    if not FB_PAGE_ACCESS_TOKEN or not INSTAGRAM_BUSINESS_ACCOUNT_ID or INSTAGRAM_BUSINESS_ACCOUNT_ID == "YOUR_INSTAGRAM_BUSINESS_ACCOUNT_ID":
        warnings.append("Instagram Graph API credentials are not fully set or placeholder ID is used. Instagram posting will fail.")

    if warnings:
        print("\n--- API KEY WARNINGS ---")
        for warning in warnings:
            print(f"- {warning}")
        print("------------------------\n")
        input("Press Enter to continue (or Ctrl+C to exit)...") # Pause execution to allow user to read warnings


# --- Main Workflow Execution ---
if __name__ == "__main__":
    check_api_keys() # Check API keys at the start

    workflow_manager = WorkflowStateManager()
    news_fetcher = NewsFetcher()
    text_processor = TextProcessor() # For Deepseek summary/title
    image_fetcher = ImageFetcher()
    image_local_processor = ImageLocalProcessor()
    caption_generator = CaptionGenerator() # NEW
    cloudinary_uploader = CloudinaryUploader() # NEW
    instagram_poster = InstagramPoster() # NEW
    local_saver = LocalSaver(IMAGE_OUTPUT_DIR, JSON_OUTPUT_DIR, EXCEL_OUTPUT_DIR, ALL_POSTS_JSON_FILE, ALL_POSTS_EXCEL_FILE) # Pass config values

    # NEW: Initialize OpenAI client for OpenRouter Mistral model for analysis
    # This client is specifically for the analysis function, other LLM calls are via TextProcessor/CaptionGenerator
    mistral_client_for_analysis = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_MISTRAL_API_KEY,
        # Removed 'headers' keyword argument as it's not supported here
    )


    try:
        # NEW: Check and run weekly analysis
        current_style_recommendations = _load_style_recommendations() # Load previous recommendations initially
        recommendation_text_for_llm = current_style_recommendations.get('weekly_analysis', '') # Default to empty string

        if workflow_manager.should_run_weekly_analysis():
            print("Time to run weekly analysis...")
            try:
                # Pass the module-level mistral_client_for_analysis instance
                new_recommendations = perform_weekly_analysis(mistral_client_for_analysis, local_saver) # Pass local_saver
                if new_recommendations:
                    current_style_recommendations = new_recommendations # Update if new recommendations were generated
                    recommendation_text_for_llm = current_style_recommendations.get('weekly_analysis', '')
                workflow_manager.update_last_analysis_timestamp() # Update timestamp regardless of recommendation success
            except Exception as e:
                print(f"Error during weekly analysis: {e}")
                import traceback
                traceback.print_exc()
                # Continue with existing recommendations if analysis fails

        if recommendation_text_for_llm:
            print(f"\nApplying current style recommendations:\n{recommendation_text_for_llm}\n")
        else:
            print("\nNo specific style recommendations to apply at this time.\n")


        # Determine which type of content to fetch for this single run
        content_type_for_this_run = workflow_manager.get_current_post_type()
        post_number_for_this_run = workflow_manager.get_current_post_number()

        print(f"\n--- Processing Post {post_number_for_this_run}/{len(CONTENT_TYPE_CYCLE)} (Type: {content_type_for_this_run.replace('_', ' ').title()}) ---")

        # Fetch only one content item
        post_to_process = news_fetcher.get_single_content_item(content_type_for_this_run)

        if not post_to_process:
            print(f"No recent news/fact available for '{content_type_for_this_run.replace('_', ' ').title()}' after all attempts. Skipping post creation for this cycle.")
            # Still save state and exit gracefully
            workflow_manager.increment_post_type_index()
            sys.exit(0)

        print(f"Original Title: {post_to_process.get('title', 'N/A')}")
        print(f"Original Description: {post_to_process.get('description', 'N/A')[:100]}...")

        post_to_process['original_description'] = post_to_process.get('description', 'N/A')

        # 1. Summarize and Enhance Text (OpenRouter - Deepseek)
        short_title, summary, text_process_success = text_processor.process_text( # Updated return values
            post_to_process.get('title', ''),
            post_to_process.get('description', ''),
            post_to_process.get('type', ''),
            style_recommendations=recommendation_text_for_llm # NEW: Pass style recommendations
        )

        if not text_process_success:
            print(f"Deepseek text processing failed for this post. Skipping post creation for this cycle.")
            workflow_manager.increment_post_type_index()
            sys.exit(0)

        post_to_process['title'] = short_title
        post_to_process['summary'] = summary
        # seo_caption and hashtags will be generated by Mistral later
        post_to_process['seo_caption'] = "" # Initialize empty
        post_to_process['hashtags'] = [] # Initialize empty

        print(f"Generated Short Title: {short_title}")
        print(f"Generated Summary: {summary}")

        # 2. Fetch Relevant Image (Pexels then Unsplash then Openverse then Pixabay)
        # Note: ImageFetcher itself does not currently take style recommendations.
        # Its prompt is derived from the title/summary which are already influenced.
        image_search_prompt = short_title + " " + summary
        print(f"Fetching image for: {image_search_prompt[:50]}...")
        fetched_pil_image = image_fetcher.fetch_image(image_search_prompt)

        # 3. Overlay Text on Image and Compose Final Post
        print("Composing final post image with overlays...")
        post_to_process['incident_label'] = None # Ensure this is reset

        # If image fetching failed, generate a placeholder image
        if fetched_pil_image is None:
            print(f"No relevant image found from any source for prompt: {image_search_prompt}. Generating placeholder image.")
            # Create a simple black placeholder image with text
            placeholder_img = Image.new('RGB', (CANVAS_WIDTH, IMAGE_DISPLAY_HEIGHT), color=(70, 70, 70))
            draw_placeholder = ImageDraw.Draw(placeholder_img)
            fallback_font = load_font(FONT_PATH_REGULAR, 50)
            text_to_draw = "IMAGE NOT AVAILABLE"
            text_bbox = draw_placeholder.textbbox((0,0), text_to_draw, font=fallback_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw_placeholder.text(((CANVAS_WIDTH - text_width) / 2, (IMAGE_DISPLAY_HEIGHT - text_height) / 2),
                                  text_to_draw, font=fallback_font, fill=(200, 200, 200))
            fetched_pil_image = placeholder_img
            post_to_process['image_status'] = 'placeholder'
        else:
            post_to_process['image_status'] = 'fetched'


        final_post_image = image_local_processor.overlay_text(fetched_pil_image, {
            'title': post_to_process['title'],
            'summary': post_to_process['summary'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'source': post_to_process.get('source', 'Unknown Source'),
            'incident_label': post_to_process.get('incident_label', None),
            'content_type_display': post_to_process.get('type')
        })
        post_to_process['final_image'] = final_post_image # Assign the composed image

        # 4. Generate Caption + Hashtags (Mistral via OpenRouter)
        print("Generating caption and hashtags with Mistral...")
        instagram_caption, instagram_hashtags, caption_success = caption_generator.generate_caption_and_hashtags(
            post_to_process['title'],
            post_to_process['summary'],
            style_recommendations=recommendation_text_for_llm # NEW: Pass style recommendations
        )

        post_to_process['seo_caption'] = instagram_caption
        post_to_process['hashtags'] = instagram_hashtags

        print(f"Generated Instagram Caption: {instagram_caption[:100]}...")
        print(f"Generated Hashtags: {', '.join(instagram_hashtags)}")

        # 5. Save All Results Locally (before Cloudinary/Instagram, to ensure data is captured)
        # This also saves the 'final_image' to disk which is needed for Cloudinary upload
        print("Saving post metadata and local image...")
        local_saver.save_post(post_to_process) # This call now populates post_to_process['Post_ID']


        # Retrieve the local image path using the now available 'Post_ID'
        post_id_for_upload = post_to_process['Post_ID']
        local_image_path_for_upload = os.path.join(IMAGE_OUTPUT_DIR, f"{post_id_for_upload}.png")


        # 6. Upload image to Cloudinary
        cloudinary_image_url = None
        if post_to_process['final_image'] and os.path.exists(local_image_path_for_upload): # Check if image exists locally
            print("Uploading image to Cloudinary...")
            cloudinary_image_url = cloudinary_uploader.upload_image(
                local_image_path_for_upload,
                public_id=post_id_for_upload, # Use the generated post_id as Cloudinary public_id
                folder="insight_pulse_posts"
            )
            post_to_process['cloudinary_url'] = cloudinary_image_url
        else:
            print("Skipping Cloudinary upload: No valid local image found at path or image not generated.")
            post_to_process['cloudinary_url'] = "N/A - Image not uploaded"

        # 7. Post to Instagram
        if cloudinary_image_url:
            print("Attempting to post to Instagram...")
            combined_caption = f"{post_to_process['seo_caption']}\n\n{' '.join(post_to_process['hashtags'])}"
            instagram_post_success = instagram_poster.post_image(cloudinary_image_url, combined_caption)
            post_to_process['instagram_posted'] = instagram_post_success
        else:
            print("Skipping Instagram post: No Cloudinary image URL available.")
            post_to_process['instagram_posted'] = False


        # Final state update and exit
        # Remove final_image from dict before saving again or passing around if not needed
        # (local_saver already handles this, but good practice before incrementing state)
        if 'final_image' in post_to_process:
            del post_to_process['final_image']

        workflow_manager.increment_post_type_index()
        # The line below was causing an error if post_to_process['content_type'] was a string.
        # It should be len(CONTENT_TYPE_CYCLE) as it's the length of the *cycle*, not the current post type.
        print(f"Successfully processed post {post_number_for_this_run}/{len(CONTENT_TYPE_CYCLE)}. State updated for next trigger.")

    except Exception as e:
        print(f"\nAn unhandled error occurred during workflow execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) # Exit with error code if unhandled exception occurs
