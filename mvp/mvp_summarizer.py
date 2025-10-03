"""
Summarization models: Abstractive (BART, T5, Pegasus) & Extractive (TextRank).
"""

import torch
import yaml
from typing import Dict, List, Optional
from pathlib import Path
import networkx as nx
import nltk
from nltk.tokenize import sent_tokenize
from transformers import (
    BartForConditionalGeneration, BartTokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    PegasusForConditionalGeneration, PegasusTokenizer
)


class AbstractiveSummarizer:
    """Abstractive summarization using transformer models (BART, T5, Pegasus)."""

    def __init__(self, model_name: str, config_path: str = "config.yaml"):
        self.model_name = model_name.lower()
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = None, None
        self._load_model()

    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _load_model(self):
        model_config = self.config['models']['summarization'].get(self.model_name)
        if not model_config:
            raise ValueError(f"Model {self.model_name} not found in config")

        model_path = model_config['name']
        if self.model_name == "bart":
            self.tokenizer = BartTokenizer.from_pretrained(model_path)
            self.model = BartForConditionalGeneration.from_pretrained(model_path)
        elif self.model_name == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        elif self.model_name == "pegasus":
            self.tokenizer = PegasusTokenizer.from_pretrained(model_path)
            self.model = PegasusForConditionalGeneration.from_pretrained(model_path)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        self.model = self.model.to(self.device).eval()

    def summarize(self, text: str, length_style: str = "medium") -> str:
        model_config = self.config['models']['summarization'][self.model_name]
        max_length, min_length = self._get_length_params(length_style, model_config)

        if self.model_name == "t5":
            text = f"summarize: {text}"

        inputs = self.tokenizer.encode(
            text, return_tensors="pt", max_length=model_config["max_length"], truncation=True
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs, max_length=max_length, min_length=min_length,
                length_penalty=2.0, num_beams=4, early_stopping=True
            )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

    def _get_length_params(self, style: str, model_config: Dict) -> tuple:
        if style == "short":
            return model_config["default_min_summary"], max(10, model_config["default_min_summary"] // 2)
        elif style == "medium":
            return model_config["default_max_summary"], model_config["default_min_summary"]
        elif style == "long":
            return model_config["default_max_summary"] * 2, model_config["default_min_summary"]
        return model_config["default_max_summary"], model_config["default_min_summary"]

    def batch_summarize(self, texts: List[str]) -> List[str]:
        model_config = self.config['models']['summarization'][self.model_name]
        inputs = self.tokenizer(
            [f"summarize: {t}" if self.model_name == "t5" else t for t in texts],
            return_tensors="pt", padding=True, truncation=True,
            max_length=model_config["max_length"]
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"], attention_mask=inputs["attention_mask"],
                max_length=model_config["default_max_summary"],
                min_length=model_config["default_min_summary"],
                num_beams=4, early_stopping=True
            )

        return [self.tokenizer.decode(s, skip_special_tokens=True).strip() for s in summary_ids]


class ExtractiveSummarizer:
    """Extractive summarization using TextRank."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.num_sentences = self.config["processing"]["extractive"]["num_sentences"]
        nltk.download("punkt", quiet=True)

    def summarize(self, text: str, num_sentences: Optional[int] = None) -> str:
        if num_sentences is None:
            num_sentences = self.num_sentences

        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text

        # Build similarity graph
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np

        vectorizer = TfidfVectorizer().fit_transform(sentences)
        similarity_matrix = (vectorizer * vectorizer.T).toarray()

        # Build graph & run TextRank
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)

        ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        top_sentences = [s for _, s in ranked[:num_sentences]]
        return " ".join(top_sentences)
