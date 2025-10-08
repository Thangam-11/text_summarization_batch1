import requests
import time

class Paraphraser:
    """
    Paraphrasing using T5 model.
    Generates multiple reworded versions of input text.
    Model: ramsrigouthamg/t5_paraphraser
    """

    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = "https://api-inference.huggingface.co/models/ramsrigouthamg/t5_paraphraser"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def _generate_paraphrases(self, text, num_return_sequences):
        """Internal helper to call Hugging Face API."""
        payload = {
            "inputs": f"paraphrase: {text}",
            "parameters": {
                "max_length": 256,
                "num_return_sequences": num_return_sequences,
                "num_beams": max(5, num_return_sequences),
                "temperature": 0.9,
                "top_p": 0.95,
                "do_sample": True
            }
        }

        # Retry logic for model loading
        for attempt in range(3):
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=90)
            if response.status_code == 503:
                print(f"⏳ Model is loading (attempt {attempt + 1}/3)... waiting 20 seconds.")
                time.sleep(20)
                continue

            if response.status_code == 200:
                results = response.json()
                if isinstance(results, list) and len(results) > 0:
                    paraphrases = [r.get("generated_text", "").strip() for r in results if r.get("generated_text")]
                    return paraphrases if paraphrases else ["⚠️ No paraphrase generated."]
                return ["⚠️ No paraphrase generated."]

            return [f"❌ API Error: {response.status_code} - {response.text}"]

        return ["⚠️ Model still loading. Please try again later."]

    def paraphrase(self, text, num_return_sequences=3):
        """
        Generate multiple paraphrased versions of input text.
        """
        if not text.strip():
            return ["⚠️ Input text is empty! Please provide valid content."]

        num_return_sequences = min(max(1, num_return_sequences), 5)

        try:
            return self._generate_paraphrases(text, num_return_sequences)
        except requests.exceptions.Timeout:
            return ["❌ Request timeout. Please try again."]
        except Exception as e:
            return [f"❌ Error: {str(e)}"]

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    HF_API_KEY = os.getenv("HF_API_KEY")

    if not HF_API_KEY:
        print("⚠️ Please set HF_API_KEY in your .env file")
    else:
        paraphraser = Paraphraser(HF_API_KEY)
        text = "Machine learning is changing the world rapidly."
        print("\n✨ Input:", text)
        print("\n🔁 Paraphrased Versions:\n")
        results = paraphraser.paraphrase(text, num_return_sequences=3)
        for idx, r in enumerate(results, 1):
            print(f"{idx}. {r}")
