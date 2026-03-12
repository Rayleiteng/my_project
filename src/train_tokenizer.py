import os
import re
from collections import Counter
from tokenizers import Tokenizer, pre_tokenizers, decoders, models, trainers, Regex, AddedToken

LAYOUT_BLACKLIST = {
    r"\quad", r"\qquad", r"\vspace", r"\hspace", r"\left", r"\right",
    r"\color", r"\displaystyle", r"\\", r"\!", r"\,", r"\;"
}

def batch_iterator(input_dir, chunk_size=1000):
    """
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".mmd"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunk = []
                    for line in f:
                        text = line.strip()
                        if text:
                            chunk.append(text)
                            if len(chunk) >= chunk_size:
                                yield chunk
                                chunk = []
                    if chunk:
                        yield chunk

def scan_for_semantic_commands(input_dir, min_freq=5):
    pattern = re.compile(r"\\[a-zA-Z]+|\\[^a-zA-Z\s]")
    command_counts = Counter()
    for chunk in batch_iterator(input_dir):
        for line in chunk:
            matches = pattern.findall(line)
            command_counts.update(matches)
            
    valid_commands = [
        cmd for cmd, count in command_counts.items() 
        if count >= min_freq and cmd not in LAYOUT_BLACKLIST
    ]
    return valid_commands

def train_industrial_math_tokenizer(input_dir, save_path, vocab_size=50000):
    semantic_commands = scan_for_semantic_commands(input_dir)
    
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    
    custom_regex = Regex(r"\\(?:begin|end)\{[a-zA-Z*]+\}|\\(?:[a-zA-Z]+|[^a-zA-Z\s])|\d+(?:\.\d+)?|[a-zA-Z]+|[^\s\w]|\^")
    
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=custom_regex, behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False)
    ])
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    
    user_defined_symbols = [AddedToken(cmd, single_word=False) for cmd in semantic_commands]
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.add_tokens(user_defined_symbols)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;':\",./<>?\\")
    )

    tokenizer.train_from_iterator((line for chunk in batch_iterator(input_dir) for line in chunk), trainer)

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    tokenizer.save(save_path)

    test_cases = [
        r"\int_{-\infty}^{\infty} \hat{f}(\xi) e^{2 \pi i \xi x} d\xi",
        r"|\psi\rangle = \alpha|0\rangle + \beta|1\rangle",
        r"x_{i, j} + y^{2}",
        r"\alpha_{i,j}"
    ]
    
    for text in test_cases:
        encoded = tokenizer.encode(text)
        print(f"\n context: {text}")
        print(f"Tokens: {encoded.tokens}")

if __name__ == "__main__":
    input_directory = "../data/processed_latex"
    
    output_model = "../data/tokenizer_math.json"
    
    
    train_industrial_math_tokenizer(input_directory, output_model, vocab_size=30000)
    
