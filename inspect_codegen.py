from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Salesforce/codegen-350M-multi",
    trust_remote_code=True
)

print("\n====== ATTENTION LAYERS FOUND ======\n")
for name, _ in model.named_modules():
    if "attn" in name.lower():
        print(name)

print("\n====== MLP LAYERS FOUND ======\n")
for name, _ in model.named_modules():
    if "mlp" in name.lower() or "ffn" in name.lower():
        print(name)
