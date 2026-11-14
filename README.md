# MSLG-SPA-2026

Code for my submission to the MSLG-SPA 2026 shared task at IberLEF 2026: bidirectional translation between Mexican Sign Language (MSL) glosses and Spanish.

## Approach

The system fine-tunes mBART-50 Large with transfer learning, since the official training set is small (489 pairs). The pipeline has four phases, described in the paper in `CEURART/`:

1. Domain adaptation on the German sign language dataset PHOENIX-2014-T (`setup_pretrain.py`, `train_phoenix.py`).
2. Synthetic pseudo-gloss data built by rule-based conversion of Spanish sentences (`glosado.py`, `train_synthetic.py`).
3. Fine-tuning on the official training data (`train_finetune.py`).
4. Inference with beam search (`run_inference.py`).

## Other scripts

- `submission_formatter.py`, `validate_submission.py`, `verify_ids.py`, `final_audit.py`: prepare and check submission files.
- `inspect_failures.py`, `test_tokenizer.py`: debugging helpers. `inspect_failures.py` contains local file paths that need to be adjusted.

## Setup

```
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

## Data

The official MSLG-SPA data and PHOENIX-2014-T are not included in this repository. Please obtain them from the task organizers and the dataset authors, under their terms.

## Paper

`CEURART/` contains the LaTeX source and PDF of the working notes paper: "MBART-50 Based Transfer Learning for Bidirectional MSL-Spanish Translation" (IberLEF 2026).
