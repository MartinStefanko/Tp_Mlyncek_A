import os
import time
from google import genai
from dotenv import load_dotenv


def get_client(api_key=None):
    load_dotenv()
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=key)


def generate_with_retry(prompt, client=None, model="gemini-2.5-flash-lite", retries=5, delay=3.0, verbose=True):
    if client is None:
        client = get_client()

    for attempt in range(1, retries + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=prompt
            )
        except Exception as e:
            if attempt == retries:
                raise
            if verbose:
                print(f"LLM zlyhalo (pokus {attempt}/{retries}), cakam {delay}s a opakujem. Chyba: {e}")
            time.sleep(delay)
