import warnings
from typing import List, Tuple, Literal

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM
from dotenv import load_dotenv
from . import config

class SpladeEncoder:
    def __init__(
        self,
        model_name: str = "naver/splade-v3",
        pooling: Literal["max", "mean"] = "max",
        use_logits: bool = True
    ):
        """
        Args:
            model_name: Checkpoint name.
            pooling: Pooling method ('max' or 'mean').
            use_logits: If True, uses outputs.logits; if False, uses hidden_states.
        """
        self.model_name = model_name
        self.pooling = pooling
        self.use_logits = use_logits

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
        self.model.eval()

        self.reverse_vocab = {idx: tok for tok, idx in self.tokenizer.get_vocab().items()}

    def get_sparse_vector(self, text: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Expands a text into a SPLADE-weighted BoW representation.

        Args:
            text: Input text.
            top_k: Number of most relevant terms to return.

        Returns:
            List of (token, weight) pairs.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)

            if self.use_logits:
                scores = outputs.logits
            else:
                # e.g.: use last layer of hidden_states
                scores = outputs.hidden_states[-1]

        activated = torch.log1p(F.relu(scores))

        if self.pooling == "max":
            doc_rep = torch.max(activated, dim=1).values.squeeze(0)
        elif self.pooling == "mean":
            doc_rep = torch.mean(activated, dim=1).squeeze(0)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        weights = doc_rep.cpu().tolist()
        non_zero_weights = {i: v for i, v in enumerate(weights) if v > 0}

        sorted_items = sorted(non_zero_weights.items(), key=lambda item: item[1], reverse=True)
        bow_rep = [
            (self.reverse_vocab.get(k, f"[UNK-{k}]"), round(v, 2))
            for k, v in sorted_items[:top_k]
        ]
        return bow_rep