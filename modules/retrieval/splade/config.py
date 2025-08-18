import os
from dotenv import load_dotenv

def setup_hf_authentication():
    """
    Loads HF token from .env or colab and log in
    """
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    if hf_token is None:
        try:
            from google.colab import userdata
            hf_token = userdata.get("HF_TOKEN")
        except ImportError:
            pass

    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        return True
    return False