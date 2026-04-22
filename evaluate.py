import sacrebleu

# Read predictions
with open("output.txt", "r", encoding="utf-8") as f:
    predictions = [line.strip() for line in f if line.strip()]

# Read references
with open("reference.txt", "r", encoding="utf-8") as f:
    references = [line.strip() for line in f if line.strip()]

# Compute BLEU
bleu = sacrebleu.corpus_bleu(predictions, [references])

print("BLEU Score:", bleu.score)
