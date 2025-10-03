import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

class SummarizerAPI:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.client = InferenceClient(api_key=HF_API_KEY)
        self.model = model_name

    def summarize(self, text: str) -> str:
        result = self.client.text_generation(
            model=self.model,
            prompt=text,
            max_new_tokens=150
        )
        return result.strip()


class ParaphraserAPI:
    def __init__(self, model_name="t5-small"):
        self.client = InferenceClient(api_key=HF_API_KEY)
        self.model = model_name

    def paraphrase(self, text: str) -> str:
        prompt = f"paraphrase: {text}"
        result = self.client.text_generation(
            model=self.model,
            prompt=prompt,
            max_new_tokens=100
        )
        return result.strip()
