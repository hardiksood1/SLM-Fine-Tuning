from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from .preprocessing import ConcatenatedDataset

def load_and_tokenize_datasets(dataset_configs, split='train', seq_length=1024, tokenizer=None):
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided")

    all_tokenized = []

    for entry in dataset_configs:
        print(f"Loading {entry['id']}...")
        ds = load_dataset(entry['id'], entry.get('config'), split=split)
        max_samples = entry.get('max_samples')
        if max_samples:
            ds = ds.select(range(min(len(ds), max_samples)))

        text_key = 'text' if 'text' in ds.features else next(iter(ds.features))

        for example in tqdm(ds, desc=f"Tokenizing {entry['id']}"):
            text = example[text_key]
            if text:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > 0:
                    all_tokenized.append(tokens)

    print(f"Total raw tokens: {sum(len(t) for t in all_tokenized)}")
    return all_tokenized