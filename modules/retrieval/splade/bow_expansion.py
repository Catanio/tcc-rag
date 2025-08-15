if __name__ == "__main__":
    from huggingface_hub import login
    login("hf_xxx")

    import warnings
    warnings.filterwarnings("ignore")

    from transformers import AutoTokenizer, AutoModelForMaskedLM
    import torch
    import torch.nn.functional as F

    # model_name = "naver/splade-cocondenser-ensembledistil"
    model_name = "naver/splade-v3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()

    # inverted vocabulary: index -> token
    reverse_vocab = {idx: tok for tok, idx in tokenizer.get_vocab().items()}

    def splade_expand_bow(text, top_k=20):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # shape: [1, seq_len, vocab_size]

        import torch.nn.functional as F

        # apply ReLU + log1p
        activated = torch.log1p(F.relu(logits))

        # max pooling
        doc_rep = torch.max(activated, dim=1).values.squeeze(0)

        # Converts to dictionary {token_id: weight} only for weights > 0
        weights = doc_rep.cpu().tolist()
        d = {i: v for i, v in enumerate(weights) if v > 0}

        # Orders and converts to legible format (token, peso)
        sorted_d = dict(sorted(d.items(), key=lambda item: item[1], reverse=True))
        bow_rep = [(reverse_vocab.get(k, f"[UNK-{k}]"), round(v, 2)) for k, v in list(sorted_d.items())[:top_k]]
        
        return bow_rep

    # example
    text = "what causes aging fast"
    bow = splade_expand_bow(text, top_k=25)

    print("SPLADE BOW rep:\n", bow)