# main.py

import requests
import json
import os
import pandas as pd
from datetime import datetime, timedelta, UTC
from PIL import Image, ImageDraw, ImageFont, ImageOps
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
    from diffusers import (
        FluxPipeline, StableDiffusion3Pipeline, DiffusionPipeline,
        StableDiffusionPipeline
    )
    import transformers
    AI_LIBRARIES_AVAILABLE = True
except ImportError:
    print("Warning: `torch`, `diffusers`, or `transformers` not found. AI image generation will be disabled.")
    AI_LIBRARIES_AVAILABLE = False

# Fix for some SSL certificate issues
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context

from config import *
from state_manager import WorkflowStateManager

# --- Utility Functions ---
def load_font(font_path, size):
    try:
        return ImageFont.truetype(font_path, size)
    except IOError:
        print(f"Error: Font file not found at {font_path}.")
        return ImageFont.load_default()

def wrap_text_by_word_count(text, font, max_width_pixels, max_words=None):
    if not text: return [""]
    words = text.split(' ')
    if max_words is not None and len(words) > max_words:
        words = words[:max_words]
        text_to_wrap = ' '.join(words) + "..."
    else:
        text_to_wrap = ' '.join(words)
    lines, current_line_words = [], []
    dummy_draw = ImageDraw.Draw(Image.new('RGB', (1,1)))
    for word in text_to_wrap.split(' '):
        test_line = ' '.join(current_line_words + [word])
        text_width = dummy_draw.textbbox((0,0), test_line, font=font)[2]
        if text_width <= max_width_pixels:
            current_line_words.append(word)
        else:
            if current_line_words: lines.append(' '.join(current_line_words))
            current_line_words = [word]
    if current_line_words: lines.append(' '.join(current_line_words))
    return lines

# --- Main Classes ---
class BackgroundGenerator:
    def generate_gradient_background(self, width, height, color1, color2):
        img = Image.new('RGBA', (width, height), color1)
        draw = ImageDraw.Draw(img)
        for y in range(height):
            r = int(color1[0] + (color2[0] - color1[0]) * (y / height))
            g = int(color1[1] + (color2[1] - color1[1]) * (y / height))
            b = int(color1[2] + (color2[2] - color1[2]) * (y / height))
            draw.line([(0, y), (width, y)], fill=(r, g, b, 255))
        return img

class NewsFetcher:
    def _fetch_from_rss(self, rss_url):
        try:
            feed = feedparser.parse(rss_url)
            if feed.bozo: print(f"Warning: RSS feed parsing issues for {rss_url}: {feed.bozo_exception}")
            recent_articles = []
            time_threshold = datetime.now(UTC) - timedelta(hours=48)
            for entry in feed.entries:
                published_dt = datetime.now(UTC)
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    published_dt = datetime(*entry.published_parsed[:6], tzinfo=UTC)
                if published_dt > time_threshold:
                    clean_desc = re.sub(r'<[^>]+>', '', entry.summary).strip()
                    recent_articles.append({'title': entry.title, 'description': clean_desc, 'url': entry.link, 'source': feed.feed.title})
                    if len(recent_articles) >= 1: break
            return recent_articles
        except Exception as e:
            print(f"Error fetching from RSS feed {rss_url}: {e}")
            return []

    def get_single_content_item(self, content_type: str):
        sources = {
            'world_news': [{'url': 'http://feeds.bbci.co.uk/news/world/rss.xml', 'name': 'BBC World News'}],
            'indian_news': [{'url': 'https://www.thehindu.com/feeder/default.rss', 'name': 'The Hindu'}],
            'tech_news': [{'url': 'https://www.theverge.com/rss/index.xml', 'name': 'The Verge'}],
            'environmental_news': [{'url': 'https://www.theguardian.com/environment/rss', 'name': 'The Guardian Environment'}]
        }
        for source in sources.get(content_type, []):
            print(f"Fetching {content_type.replace('_', ' ').title()} from: {source['name']}...")
            articles = self._fetch_from_rss(source['url'])
            if articles:
                article = articles[0]
                return {'type': content_type, **article}
        print(f"No recent articles found for {content_type}.")
        return None

class TextProcessor:
    def __init__(self):
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

    def process_text(self, title, description, post_type):
        messages = [
            {"role": "system", "content": "You are an expert content summarizer. Generate a concise headline and a clear summary from the provided news content. The output MUST be a valid JSON object with keys 'short_title' and 'summary_text'."},
            {"role": "user", "content": f"Content Type: {post_type}\nOriginal Title: {title}\nOriginal Description: {description}\n\nReturn ONLY the JSON object."}
        ]
        try:
            completion = self.client.chat.completions.create(model=OPENROUTER_MODEL, messages=messages, response_format={"type": "json_object"})
            data = json.loads(completion.choices[0].message.content)
            return data.get('short_title'), data.get('summary_text'), True
        except Exception as e:
            print(f"Error in TextProcessor: {e}")
            return ' '.join(title.split()[:4]), ' '.join(description.split()[:50]), False

class CaptionGenerator:
    def __init__(self):
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_MISTRAL_API_KEY)

    def generate_caption_and_hashtags(self, short_title, summary):
        messages = [
            {"role": "system", "content": "You are a creative social media manager. Generate an engaging Instagram caption and 10 relevant hashtags. The output MUST be a valid JSON object with keys 'caption' and 'hashtags'."},
            {"role": "user", "content": f"News Title: {short_title}\nNews Summary: {summary}\n\nReturn ONLY the JSON object."}
        ]
        try:
            completion = self.client.chat.completions.create(model=OPENROUTER_MISTRAL_MODEL, messages=messages, response_format={"type": "json_object"})
            data = json.loads(completion.choices[0].message.content)
            return data.get('caption'), data.get('hashtags', []), True
        except Exception as e:
            print(f"Error in CaptionGenerator: {e}")
            return "Follow for more updates.", ["#news"], False

    def generate_image_prompt(self, short_title, summary):
        messages = [
            {"role": "system", "content": "You are an expert in creating image generation prompts. Generate a descriptive, photorealistic prompt for a news story. The prompt MUST NOT contain any text or letters. Return ONLY the prompt string."},
            {"role": "user", "content": f"News Title: {short_title}\nNews Summary: {summary}\n\nReturn ONLY the prompt string."}
        ]
        try:
            completion = self.client.chat.completions.create(model=OPENROUTER_MISTRAL_MODEL, messages=messages)
            return completion.choices[0].message.content.strip().replace('"', ''), True
        except Exception as e:
            print(f"Error generating image prompt: {e}")
            return f"Symbolic image for {short_title}", False

class HuggingFaceImageGenerator:
    """Tries a sequence of Hugging Face models to generate an image."""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        if not HUGGING_FACE_TOKEN or "YOUR_" in HUGGING_FACE_TOKEN:
            print("HUGGING_FACE_TOKEN is not set. AI image generation will be disabled.")
            self.token = None
        else:
            self.token = HUGGING_FACE_TOKEN

    def generate_image(self, prompt):
        if not self.token or not AI_LIBRARIES_AVAILABLE:
            return None

        for model_id in AI_IMAGE_MODELS_TO_TRY:
            print(f"\n--- Attempting to generate image with model: {model_id} ---")
            try:
                pipe = self._get_pipeline(model_id)
                if pipe:
                    pipe.to(self.device)
                    # For resource-constrained environments like GitHub Actions runners
                    if self.device == "cpu":
                        pipe.enable_model_cpu_offload()

                    image = self._run_pipeline(pipe, prompt, model_id)
                    if image:
                        print(f"Successfully generated image with {model_id}.")
                        return image
            except Exception as e:
                print(f"Failed to generate image with {model_id}. Error: {e}")
                traceback.print_exc()
                # Clean up memory before trying the next model
                if 'pipe' in locals():
                    del pipe
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print("All AI image generation models failed.")
        return None

    def _get_pipeline(self, model_id):
        """Loads the correct pipeline based on the model ID."""
        common_args = {
            "torch_dtype": self.torch_dtype,
            "token": self.token
        }
        if "flux" in model_id.lower():
            return FluxPipeline.from_pretrained(model_id, **common_args)
        if "stable-diffusion-3" in model_id.lower():
            return StableDiffusion3Pipeline.from_pretrained(model_id, **common_args)
        if "stable-diffusion-xl" in model_id.lower():
            return DiffusionPipeline.from_pretrained(model_id, use_safetensors=True, variant="fp16", **common_args)
        if "stable-diffusion" in model_id.lower():
            return StableDiffusionPipeline.from_pretrained(model_id, **common_args)
        print(f"Unknown pipeline for model: {model_id}")
        return None

    def _run_pipeline(self, pipe, prompt, model_id):
        """Runs the generation with parameters tailored for the model."""
        generator = torch.Generator(device="cpu").manual_seed(random.randint(0, 2**32 - 1))
        if "flux" in model_id.lower():
            return pipe(prompt, num_inference_steps=50, generator=generator).images[0]
        else: # Default for Stable Diffusion models
            return pipe(prompt, num_inference_steps=28, guidance_scale=7.0, generator=generator).images[0]

class ImageFetcher:
    """Fetches images from Pexels as a fallback."""
    def fetch_image(self, prompt):
        try:
            if not PEXELS_API_KEY or "YOUR_" in PEXELS_API_KEY: return None
            headers = {"Authorization": PEXELS_API_KEY}
            params = {"query": prompt, "orientation": "portrait", "per_page": 1}
            response = requests.get(PEXELS_API_URL, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            if data and data['photos']:
                img_data = requests.get(data['photos'][0]['src']['original'], stream=True, timeout=15).content
                return Image.open(io.BytesIO(img_data))
        except Exception as e:
            print(f"Error fetching from Pexels: {e}")
        return None

class ImageLocalProcessor:
    """Handles local image processing and text overlays."""
    def overlay_text(self, base_pil_image, post_data):
        final_canvas = BackgroundGenerator().generate_gradient_background(CANVAS_WIDTH, CANVAS_HEIGHT, COLOR_GRADIENT_TOP_LEFT, COLOR_GRADIENT_BOTTOM_RIGHT)
        draw = ImageDraw.Draw(final_canvas)
        
        # Resize and paste image
        base_pil_image = ImageOps.fit(base_pil_image, (IMAGE_DISPLAY_WIDTH, IMAGE_DISPLAY_HEIGHT), Image.Resampling.LANCZOS)
        img_x = (CANVAS_WIDTH - IMAGE_DISPLAY_WIDTH) // 2
        img_y = 150
        final_canvas.paste(base_pil_image, (img_x, img_y))

        # Draw text elements
        font_headline = load_font(FONT_PATH_EXTRABOLD, FONT_SIZE_HEADLINE)
        font_summary = load_font(FONT_PATH_REGULAR, FONT_SIZE_SUMMARY)
        wrapped_title = wrap_text_by_word_count(post_data.get('title', ''), font_headline, IMAGE_DISPLAY_WIDTH, max_words=TITLE_MAX_WORDS)
        wrapped_summary = wrap_text_by_word_count(post_data.get('summary', ''), font_summary, IMAGE_DISPLAY_WIDTH, max_words=SUMMARY_MAX_WORDS)

        current_y = img_y + IMAGE_DISPLAY_HEIGHT + TITLE_TOP_MARGIN_FROM_IMAGE
        for line in wrapped_title:
            line_width = draw.textlength(line, font=font_headline)
            draw.text(((CANVAS_WIDTH - line_width) / 2, current_y), line, font=font_headline, fill=COLOR_WHITE)
            current_y += font_headline.getbbox(line)[3] + TITLE_LINE_SPACING

        current_y += SUMMARY_TOP_MARGIN_FROM_TITLE
        for line in wrapped_summary:
            draw.text((LEFT_PADDING, current_y), line, font=font_summary, fill=COLOR_WHITE)
            current_y += font_summary.getbbox(line)[3] + SUMMARY_LINE_SPACING

        return final_canvas

class CloudinaryUploader:
    def __init__(self):
        cloudinary.config(cloud_name=CLOUDINARY_CLOUD_NAME, api_key=CLOUDINARY_API_KEY, api_secret=CLOUDINARY_API_SECRET)
    def upload_image(self, image_path, public_id):
        try:
            result = cloudinary.uploader.upload(image_path, public_id=public_id, folder="insight_pulse_posts")
            return result.get('secure_url')
        except Exception as e:
            print(f"Cloudinary upload error: {e}")
            return None

class InstagramPoster:
    def __init__(self):
        self.base_url = f"https://graph.facebook.com/v19.0/{INSTAGRAM_BUSINESS_ACCOUNT_ID}"
    def post_image(self, image_url, caption):
        if not all([image_url, FB_PAGE_ACCESS_TOKEN, INSTAGRAM_BUSINESS_ACCOUNT_ID]): return False
        try:
            media_params = {'image_url': image_url, 'caption': caption, 'access_token': FB_PAGE_ACCESS_TOKEN}
            media_response = requests.post(f"{self.base_url}/media", params=media_params).json()
            if 'id' in media_response:
                publish_params = {'creation_id': media_response['id'], 'access_token': FB_PAGE_ACCESS_TOKEN}
                publish_response = requests.post(f"{self.base_url}/media_publish", params=publish_params).json()
                return 'id' in publish_response
        except Exception as e:
            print(f"Instagram post error: {e}")
        return False

class LocalSaver:
    def __init__(self):
        os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
        os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
    def save_post(self, post_data, workflow_manager):
        post_id = f"{post_data.get('type')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_post-{workflow_manager.get_current_post_number()}"
        post_data['Post_ID'] = post_id
        img_path = os.path.join(IMAGE_OUTPUT_DIR, f"{post_id}.png")
        if 'final_image' in post_data: post_data['final_image'].save(img_path)
        
        # Save metadata to JSON, excluding the image object
        metadata = {k: v for k, v in post_data.items() if k != 'final_image'}
        all_posts = []
        if os.path.exists(ALL_POSTS_JSON_FILE):
            with open(ALL_POSTS_JSON_FILE, 'r') as f:
                try: all_posts = json.load(f)
                except json.JSONDecodeError: pass
        all_posts.append(metadata)
        with open(ALL_POSTS_JSON_FILE, 'w') as f:
            json.dump(all_posts, f, indent=4)

# --- Main Workflow ---
if __name__ == "__main__":
    workflow_manager = WorkflowStateManager()
    news_fetcher = NewsFetcher()
    text_processor = TextProcessor()
    caption_generator = CaptionGenerator()
    ai_image_gen = HuggingFaceImageGenerator()
    image_fetcher = ImageFetcher()
    image_local_processor = ImageLocalProcessor()
    cloudinary_uploader = CloudinaryUploader()
    instagram_poster = InstagramPoster()
    local_saver = LocalSaver()

    try:
        content_type = workflow_manager.get_current_post_type()
        print(f"\n--- Processing Post (Type: {content_type}) ---")
        post_data = news_fetcher.get_single_content_item(content_type)
        if not post_data:
            sys.exit(f"No content found for {content_type}.")

        title, summary, _ = text_processor.process_text(post_data['title'], post_data['description'], post_data['type'])
        post_data.update({'title': title, 'summary': summary})
        print(f"Generated Title: {title}\nGenerated Summary: {summary}")

        pil_image = None
        if ENABLE_AI_IMAGE_GENERATION:
            prompt, _ = caption_generator.generate_image_prompt(title, summary)
            pil_image = ai_image_gen.generate_image(prompt)
            post_data['image_status'] = 'generated_ai' if pil_image else 'ai_failed'

        if not pil_image:
            print("AI generation failed. Falling back to stock photo API.")
            pil_image = image_fetcher.fetch_image(f"{title} {summary}")
            post_data['image_status'] = 'fetched_api' if pil_image else 'api_failed'

        if not pil_image:
            print("All image sources failed. Creating placeholder.")
            pil_image = Image.new('RGB', (IMAGE_DISPLAY_WIDTH, IMAGE_DISPLAY_HEIGHT), color='black')
            post_data['image_status'] = 'placeholder'
        
        post_data['final_image'] = image_local_processor.overlay_text(pil_image, post_data)
        
        caption, hashtags, _ = caption_generator.generate_caption_and_hashtags(title, summary)
        post_data.update({'seo_caption': caption, 'hashtags': hashtags})

        local_saver.save_post(post_data, workflow_manager)
        
        img_path = os.path.join(IMAGE_OUTPUT_DIR, f"{post_data['Post_ID']}.png")
        cloudinary_url = cloudinary_uploader.upload_image(img_path, post_data['Post_ID'])
        
        if cloudinary_url:
            full_caption = f"{caption}\n\n{' '.join(hashtags)}"
            instagram_poster.post_image(cloudinary_url, full_caption)

        workflow_manager.increment_post_type_index()
        print("--- Workflow finished successfully ---")

    except Exception as e:
        print(f"\nAn unhandled error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)
