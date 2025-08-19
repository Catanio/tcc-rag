import argparse
from .splade_encoder import SpladeEncoder
from . import config

def main():
    parser = argparse.ArgumentParser(description="Run SPLADE PoC")
    parser.add_argument("--model", type=str, default="naver/splade-v3")
    args = parser.parse_args()

    config.setup_hf_authentication()
    encoder = SpladeEncoder(model_name=args.model)
    bow = encoder.get_sparse_vector("Whats the best way to expand a query", top_k=10)
    print(bow)

if __name__ == "__main__":
    main()