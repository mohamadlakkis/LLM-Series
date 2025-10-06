import re
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "/Users/mohammadzbeeb/LiquidAI/LFM2-350M"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="bfloat16",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

ds = load_dataset("gsm8k", "main", split="train[:10]")

def extract_gold(ans: str):
    m = re.findall(r"####\s*([-\d,\.]+)", ans)
    return m[-1].replace(",", "") if m else ""

def extract_number(text: str):
    m = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    return m[-1] if m else ""

def solve(question: str):
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Solve this and give only the final numeric answer.\n" + question}],
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    ).to(model.device)

    output = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.3,
        min_p=0.15,
        repetition_penalty=1.05,
        max_new_tokens=128,
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return extract_number(text)

correct = 0
for i, ex in enumerate(ds):
    gold = extract_gold(ex["answer"])
    pred = solve(ex["question"])
    print(f"Q{i+1} pred: {pred}, gold: {gold}")
    if pred == gold:
        correct += 1

print(f"\nAccuracy on first 10 GSM8K examples: {correct/len(ds):.2%}")
