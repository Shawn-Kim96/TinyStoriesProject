import argparse
from src.bpe_tokenizer import BPETokenizerWrapper
from src.dataset import simple_tokenize

def compare_tokenization(text, bpe_model_name="gpt2"):
    """
    Compare the tokenization from a simple tokenizer and a BPE tokenizer.
    
    Args:
        text: The text to tokenize
        bpe_model_name: The name of the pre-trained BPE tokenizer to use
    """
    print("Original text:")
    print(f'"{text}"')
    print("\n" + "-" * 50 + "\n")
    
    # Simple tokenization
    simple_tokens = simple_tokenize(text)
    print(f"Simple tokenization ({len(simple_tokens)} tokens):")
    print(simple_tokens)
    print("\n" + "-" * 50 + "\n")
    
    # BPE tokenization
    bpe_tokenizer = BPETokenizerWrapper(model_name=bpe_model_name)
    bpe_encoding = bpe_tokenizer.tokenizer.encode(text, add_special_tokens=False)
    bpe_tokens = bpe_tokenizer.tokenizer.convert_ids_to_tokens(bpe_encoding)
    
    print(f"BPE tokenization with {bpe_model_name} ({len(bpe_tokens)} tokens):")
    print(bpe_tokens)
    print("\n" + "-" * 50 + "\n")
    
    # Print token IDs
    print(f"BPE token IDs:")
    print(bpe_encoding)
    print("\n" + "-" * 50 + "\n")
    
    # Try different BPE models
    if bpe_model_name == "gpt2":
        alt_models = ["distilbert-base-uncased", "roberta-base"]
        for model in alt_models:
            alt_tokenizer = BPETokenizerWrapper(model_name=model)
            alt_encoding = alt_tokenizer.tokenizer.encode(text, add_special_tokens=False)
            alt_tokens = alt_tokenizer.tokenizer.convert_ids_to_tokens(alt_encoding)
            
            print(f"{model} tokenization ({len(alt_tokens)} tokens):")
            print(alt_tokens)
            print("\n" + "-" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare simple tokenization with BPE tokenization")
    parser.add_argument("--text", type=str, default="Once upon a time, there was a little girl named Lily who loved to play with toys.", 
                       help="The text to tokenize")
    parser.add_argument("--model", type=str, default="gpt2", 
                       help="The pre-trained model to use for BPE tokenization")
    
    args = parser.parse_args()
    compare_tokenization(args.text, args.model)


if __name__ == "__main__":
    # Example usage
    main() 