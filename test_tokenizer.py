from transformers import MBart50TokenizerFast
import os
import sys

# Set IO to UTF-8 to avoid charmap errors on Windows
sys.stdout.reconfigure(encoding='utf-8')

model_path = 'f:/!Projects/MSLG-SPA 2026/mBART-50-LSM'
if not os.path.exists(model_path):
    model_path = 'f:/!Projects/MSLG-SPA 2026/mBART-50'

tokenizer = MBart50TokenizerFast.from_pretrained(model_path)

samples = ['dm-LUIS', 'MAMÁ+PAPÁ', '#OK', 'YA-VEO', 'PONER-ATENCIÓN']
print(f"Testing tokenizer from: {model_path}")
print("-" * 30)

for s in samples:
    tokens = tokenizer.tokenize(s)
    ids = tokenizer.encode(s, add_special_tokens=False)
    print(f"STRING: {s}")
    print(f"TOKENS: {tokens}")
    print(f"IDS:    {ids}")
    print("-" * 30)

# Check if these symbols exist or get split
symbols = ['-', '+', '#', 'dm-']
for sym in symbols:
    print(f"SYMBOL '{sym}': {tokenizer.tokenize(sym)}")
