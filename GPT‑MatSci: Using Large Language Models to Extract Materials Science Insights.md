# GPT‑MatSci: Using Large Language Models to Extract Materials Science Insights
**Author:** Kushagra Sharma  
**Date:** 2025-06-01  
**Status:** Working paper (GitHub preprint)  


## Abstract
Materials research produces immense unstructured text—papers, reports, and patents—with critical facts about compositions, processing, and device performance. This working paper presents an open pipeline that uses large language models (LLMs) to (i) extract structured records from literature, (ii) build task‑specific knowledge graphs, and (iii) support downstream analysis and prediction. We emphasize transparent prompts, evaluation against annotated subsets, and careful human‑in‑the‑loop validation.

## Keywords
large language models; information extraction; knowledge graph; materials informatics; literature mining

## 1. Problem framing
Key device‑level insights in materials science are locked in text. Manual extraction is slow and inconsistent, leading to stale databases. Recent studies show LLMs, when tuned and evaluated carefully, can recover device‑level entities and relations at high quality and even support preliminary performance prediction without hand‑engineered features. This repo packages those ideas in a reproducible way for domain users.

## 2. Method overview
- **Corpus builder.** Utilities to assemble topic‑specific corpora (e.g., dielectric polymers, perovskite solar cells) from open sources or user‑provided PDFs; de‑duplication and metadata normalization.

- **Schema.** Extensible JSON schema for entities (material, composition, processing step, condition, property, device metric) and relations.

- **Extraction.** Prompted or fine‑tuned LLMs (with retrieval where allowed) to populate schema; confidence scoring and rule‑based validators (units, ranges).

- **Evaluation.** Holdout annotations; entity/relation F1; ablations for prompt variants and model sizes.

- **Knowledge graph.** Build graph views for hypothesis generation and simple link prediction baselines.


## 3. Reproducibility and governance
- Deterministic runs via seed control; logging of prompts, model versions, and hashes of documents processed.
- Clear separation of training vs. evaluation corpora to prevent leakage.
- Ethical use guide: respect publisher terms; prefer open‑access; store only derived, non‑verbatim facts.


## 4. Example study (perovskite devices)
We include a miniature, open‑access subset to demonstrate Structured Information Inference and show how extracted records enable quick analytics (e.g., efficiency vs. architecture). The examples are illustrative and meant to be replaced with a team’s own curated corpus.

## 5. Suggested citation
> Sharma, K. (2025). *GPT‑MatSci: Using Large Language Models to Extract Materials Science Insights*. GitHub preprint. https://github.com/kushagrasharma/LLM-materials-knowledge

## 6. References
1. Xie, T. *et al.* “Large Language Models as Master Key: Unlocking the Secrets of Materials Science with GPT,” arXiv:2304.02213 (2023). https://arxiv.org/abs/2304.02213

2. Liu, X. *et al.* “Perovskite‑LLM: Knowledge‑Enhanced Large Language Models for Perovskite Solar Cell Research,” arXiv:2502.12669 (2025). https://arxiv.org/abs/2502.12669


## License
This document and code are released under **CC BY 4.0** / MIT (as applicable). See `LICENSE` files in the repo.
