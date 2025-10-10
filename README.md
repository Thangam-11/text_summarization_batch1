# AbstractiveSummarizer.py
import requests

class AbstractiveSummarizer:
    """
    Abstractive summarization using BART model.
    Generates new sentences that capture the meaning of the original text.
    """
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def summarize(self, text, length='medium'):
        """
        Generate abstractive summary from text.
        
        Args:
            text (str): Input text to summarize
            length (str): 'short', 'medium', or 'long'
            
        Returns:
            str: Generated summary
        """
        length_map = {
            'short': {"max_length": 60, "min_length": 30},
            'medium': {"max_length": 130, "min_length": 60},
            'long': {"max_length": 200, "min_length": 130}
        }
        
        params = length_map.get(length, length_map['medium'])
        payload = {
            "inputs": text,
            "parameters": {
                **params,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("summary_text", "No summary generated")
                else:
                    return str(result)
            elif response.status_code == 503:
                return "⚠️ Model is loading. Please try again in a few moments."
            else:
                return f"❌ API Error: {response.status_code} - {response.text}"
        except requests.exceptions.Timeout:
            return "❌ Request timeout. Please try again."
        except Exception as e:
            return f"❌ Error: {str(e)}"

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    HF_API_KEY = os.getenv("HF_API_KEY")

    if not HF_API_KEY:
        print("⚠️ Please set your Hugging Face API key in the environment variable 'HF_API_KEY'")
    else:
        print("\n🚀 Testing Abstractive Summarizer...\n")
        
        summarizer = AbstractiveSummarizer(HF_API_KEY)

        text = """
        Artificial Intelligence (AI) is revolutionizing industries by automating repetitive tasks,
        improving decision-making, and enhancing human creativity. From healthcare and education to
        finance and transportation, AI-driven solutions are reshaping how we live and work.
        """

        print("📄 Original Text:\n", text)
        print("\n🧠 Abstractive Summary:\n")



#  ExtractiveSummarizer.py



import requests

class ExtractiveSummarizer:
    """
    Extractive summarization using BART model.
    Selects and extracts important sentences from the original text.
    """
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def summarize(self, text, length='medium'):
        """
        Generate extractive summary from text.
        
        Args:
            text (str): Input text to summarize
            length (str): 'short', 'medium', or 'long'
            
        Returns:
            str: Extracted summary
        """
        length_map = {
            'short': {"max_length": 60, "min_length": 30},
            'medium': {"max_length": 130, "min_length": 60},
            'long': {"max_length": 200, "min_length": 130}
        }
        
        params = length_map.get(length, length_map['medium'])
        payload = {
            "inputs": text,
            "parameters": {
                **params,
                "do_sample": False
            }
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("summary_text", "No summary generated")
                else:
                    return str(result)
            elif response.status_code == 503:
                return "⚠️ Model is loading. Please try again in a few moments."
            else:
                return f"❌ API Error: {response.status_code} - {response.text}"
        except requests.exceptions.Timeout:
            return "❌ Request timeout. Please try again."
        except Exception as e:
            return f"❌ Error: {str(e)}"
        

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()
    HF_API_KEY = os.getenv("HF_API_KEY")

    if not HF_API_KEY:
        print("⚠️ Please set your Hugging Face API key in the environment variable 'HF_API_KEY'")
    else:
        print("\n🚀 Testing Extractive Summarizer...\n")

        summarizer = ExtractiveSummarizer(HF_API_KEY)

        text = """
        Artificial Intelligence (AI) is transforming industries by automating tasks,
        improving efficiency, and enabling innovative solutions across sectors like healthcare,
        education, and transportation. Machine learning algorithms allow computers to learn from
        data and make predictions or decisions without being explicitly programmed.
        """

        print("📄 Original Text:\n", text)
        print("\n🧠 Extractive Summary:\n")

        summary = summarizer.summarize(text, length='medium')
        print(summary)




# Paraphraser.py




import os
import requests
from dotenv import load_dotenv

class Paraphraser:
    """
    Paraphrasing using GROQ API with LLaMA 3.1 models.
    Recommended models:
      - llama-3.1-8b-instant (fast)
      - llama-3.1-70b-versatile (high quality)
    """

    def __init__(self, model_name="llama-3.1-8b-instant"):
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("❌ GROQ_API_KEY not found in .env")

        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.model_name = model_name

    def paraphrase(self, text, num_return_sequences=3):
        """
        Generate paraphrased versions of input text using GROQ API.
        """
        if not text.strip():
            return ["⚠️ Please provide valid text."]

        prompt = (
            f"Paraphrase the following text in natural English. "
            f"Provide {num_return_sequences} unique variations:\n\n{text}"
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful AI that paraphrases text naturally and clearly."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.9,
            "max_tokens": 400
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)

            if response.status_code == 200:
                data = response.json()
                text_response = data["choices"][0]["message"]["content"]
                lines = [line.strip() for line in text_response.split("\n") if line.strip()]
                return lines[:num_return_sequences]
            else:
                return [f"❌ API Error {response.status_code}: {response.text}"]

        except Exception as e:
            return [f"❌ Error: {str(e)}"]


if __name__ == "__main__":
    paraphraser = Paraphraser(model_name="llama-3.1-8b-instant")
    text = "Machine learning is changing the world rapidly."

    print(f"\n✨ Input: {text}")
    print(f"\n🤖 Using Model: {paraphraser.model_name}\n")

    results = paraphraser.paraphrase(text, num_return_sequences=3)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r}")


from mvp.extractive import ExtractiveSummarizer
from mvp.abstractive import AbstractiveSummarizer
from mvp.parapharsing import Paraphraser
import os, requests
from dotenv import load_dotenv





#  Combined Pipeline (Summarization + Paraphrasing)

class SummarizationPipeline:
    """
    Combined pipeline for Summarization (HF) + Paraphrasing (GROQ LLM).
    """

    def __init__(self, hf_api_key):
        print("🔧 Initializing SummarizationPipeline...")

        # --- Extractive Summarizer ---
        try:
            self.extractive = ExtractiveSummarizer(hf_api_key)
            print("✅ Extractive Summarizer loaded")
        except Exception as e:
            print(f"⚠️ Warning: Extractive Summarizer failed: {e}")
            self.extractive = None

        # --- Abstractive Summarizer ---
        try:
            self.abstractive = AbstractiveSummarizer(hf_api_key)
            print("✅ Abstractive Summarizer loaded")
        except Exception as e:
            print(f"⚠️ Warning: Abstractive Summarizer failed: {e}")
            self.abstractive = None

        # --- GROQ Paraphraser ---
        try:
            self.paraphraser = Paraphraser()
            print("✅ GROQ Paraphraser loaded")
        except Exception as e:
            print(f"⚠️ Warning: GROQ Paraphraser failed: {e}")
            self.paraphraser = None

        print("✨ SummarizationPipeline initialized successfully!\n")

    # -------- Summarization --------
    def summarize(self, text, method="abstractive", length="medium"):
        if not text or not text.strip():
            return "⚠️ No text provided."
        try:
            if method == "extractive":
                if self.extractive is None:
                    return "❌ Extractive Summarizer unavailable."
                return self.extractive.summarize(text, length)
            else:
                if self.abstractive is None:
                    return "❌ Abstractive Summarizer unavailable."
                return self.abstractive.summarize(text, length)
        except Exception as e:
            return f"❌ Error: {e}"

    # -------- Paraphrasing --------
    def paraphrase(self, text, num_return_sequences=3):
        if self.paraphraser is None:
            return "❌ Paraphraser unavailable."
        try:
            results = self.paraphraser.paraphrase(text, num_return_sequences)
            return "\n\n".join(results)
        except Exception as e:
            return f"❌ Error in paraphrasing: {e}"

    # -------- Utilities --------
    def get_status(self):
        return {
            "extractive": self.extractive is not None,
            "abstractive": self.abstractive is not None,
            "groq_paraphraser": self.paraphraser is not None,
        }


#  Test Runner
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    HF_API_KEY = os.getenv("HF_API_KEY")

    if not HF_API_KEY:
        print("⚠️ Please set HF_API_KEY in .env file.")
    else:
        print("🚀 Starting Summarization + Paraphrasing Test...\n")
        pipeline = SummarizationPipeline(HF_API_KEY)

        text = """
        Artificial Intelligence (AI) is transforming industries by automating repetitive tasks,
        improving efficiency, and enabling better decision-making across sectors such as healthcare,
        finance, and transportation.
        """

        # Abstractive Summary
        print("\n🧠 Abstractive Summary:\n", pipeline.summarize(text, method="abstractive"))

        # Extractive Summary
        print("\n📄 Extractive Summary:\n", pipeline.summarize(text, method="extractive"))

        # GROQ Paraphrasing
        print("\n✨ GROQ Paraphrasing Results:\n",
              pipeline.paraphrase("AI is transforming the world rapidly.", 3))

        # Status
        print("\n🔍 Module Status:\n", pipeline.get_status())

# test scripts


import os
from dotenv import load_dotenv
from mvp.extractive import ExtractiveSummarizer
from mvp.abstractive import AbstractiveSummarizer
from mvp.parapharsing import Paraphraser
from mvp.mvp_pipeline import SummarizationPipeline
import time

# Load Hugging Face API key
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

if not HF_API_KEY:
    print("⚠️ Please set your Hugging Face API key in the environment variable 'HF_API_KEY'")
    exit()

# ----------------------------------------------------------------
print("\n🚀 Starting Model Testing Suite...\n")
time.sleep(1)

# Sample text for testing
text = """
Artificial Intelligence (AI) is transforming industries by automating repetitive tasks,
improving efficiency, and enabling better decision-making across sectors such as healthcare,
finance, and transportation.
"""

# ----------------------------------------------------------------
print("🔹 Testing Individual Models\n")

# 1️⃣ Test Extractive Summarizer
extractive = ExtractiveSummarizer(HF_API_KEY)
print("🧩 Testing Extractive Summarizer...")
extractive_summary = extractive.summarize(text, length='medium')
print("📄 Extractive Summary:\n", extractive_summary, "\n")

time.sleep(2)

# 2️⃣ Test Abstractive Summarizer
abstractive = AbstractiveSummarizer(HF_API_KEY)
print("🧠 Testing Abstractive Summarizer...")
abstractive_summary = abstractive.summarize(text, length='medium')
print("📝 Abstractive Summary:\n", abstractive_summary, "\n")

time.sleep(2)

# 3️⃣ Test Paraphraser
paraphraser = Paraphraser(HF_API_KEY)
print("✨ Testing Paraphraser...")
para_results = paraphraser.paraphrase("AI is changing the world rapidly.", num_return_sequences=3)
for idx, p in enumerate(para_results, 1):
    print(f"{idx}. {p}")
print()

# ----------------------------------------------------------------
print("🔹 Testing Combined Pipeline\n")

pipeline = SummarizationPipeline(HF_API_KEY)

# Test summarization via pipeline
print("\n🧠 Abstractive Summary via Pipeline:\n", pipeline.summarize(text, method='abstractive'))
print("\n📄 Extractive Summary via Pipeline:\n", pipeline.summarize(text, method='extractive'))

# Test paraphrasing via pipeline
print("\n✨ Paraphrasing via Pipeline:\n", pipeline.paraphrase("AI is transforming industries quickly.", 2))

# Check module status
print("\n🔍 Module Status:\n", pipeline.get_status())

print("\n✅ All tests completed successfully!\n")



        summary = summarizer.summarize(text, length='medium')
        print(summary)
