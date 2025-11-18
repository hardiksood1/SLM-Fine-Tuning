import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import time
import wandb

@torch.no_grad()
def compute_perplexity(model, tokenizer, dataset_name="wikitext", split="validation", max_samples=1000):
    dataset = load_dataset(dataset_name, "wikitext-103-raw-v1", split=split)
    if max_samples:
        dataset = dataset.select(range(min(len(dataset), max_samples)))

    model.eval()
    losses = []
    for example in tqdm(dataset, desc="Evaluating"):
        text = example['text']
        if not text:
            continue
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        outputs = model(**inputs, labels=inputs["input_ids"])
        losses.append(outputs.loss.item())
    ppl = np.exp(np.mean(losses))
    return ppl

def measure_throughput(model, tokenizer, seq_length=1024, num_runs=10):
    dummy = torch.randint(0, 50257, (1, seq_length)).to(model.device)
    model.eval()
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = model.generate(dummy, max_new_tokens=50, do_sample=False)
        times.append(time.time() - start)
    return 50 / np.mean(times)

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("final_model", torch_dtype=torch.bfloat16, device_map="auto")
    if os.path.exists("final_model/adapter_config.json"):
        model = PeftModel.from_pretrained(model, "final_model")
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi")

    ppl = compute_perplexity(model, tokenizer)
    throughput = measure_throughput(model, tokenizer)

    print(f"Perplexity: {ppl:.2f}")
    print(f"Throughput: {throughput:.2f} tokens/sec")

    wandb.log({"eval/perplexity": ppl, "eval/throughput_tokens_per_sec": throughput})