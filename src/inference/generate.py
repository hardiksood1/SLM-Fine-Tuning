import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time

def generate_text(model, tokenizer, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id
    )
    elapsed = time.time() - start
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    tokens_per_sec = max_new_tokens / elapsed
    return generated, tokens_per_sec

# Example usage
if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("final_model", torch_dtype=torch.bfloat16, device_map="auto")
    if os.path.exists("final_model/adapter_config.json"):
        model = PeftModel.from_pretrained(model, "final_model")
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi")
    text, speed = generate_text(model, tokenizer, "Once upon a time")
    print(text)
    print(f"Speed: {speed:.2f} tokens/sec")