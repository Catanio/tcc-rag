import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List

class Translator:
    def __init__(self, model_name: str, batch_size: int = 32):
        
        print(f"Starting translation with model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device detected: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval() # Set model to evaluation mode (important for consistency)
        self.batch_size = batch_size

    def translate_batch(self, 
                        texts: List[str], 
                        src_lang: str = "por_Latn", 
                        tgt_lang: str = "eng_Latn",
                        max_length: int = 512) -> List[str]:

        if not texts:
            return []

        self.tokenizer.src_lang = src_lang

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)

        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        # Generate translation without calculating gradients to save memory and be faster
        with torch.no_grad():
            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                num_beams=5,  # Beam search for better quality
                max_length=512
            )

        # Decode tokens to text
        translated_texts = self.tokenizer.batch_decode(
            translated_tokens, 
            skip_special_tokens=True
        )
        return translated_texts
