from transformers import MarianMTModel, MarianTokenizer

# Load model and tokenizer
model_name = "Helsinki-NLP/opus-mt-bn-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# File paths
input_file = "input.txt"
output_file = "output.txt"

# Read input sentences
with open(input_file, "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f.readlines() if line.strip()]

translations = []

# Translate sentences
for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(
        **inputs, num_beams=5, max_length=128, early_stopping=True
    )
    tgt_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    translations.append(tgt_text)

# Write output
with open(output_file, "w", encoding="utf-8") as f:
    for line in translations:
        f.write(line + "\n")

print("✅ Translation completed. Check output.txt")
