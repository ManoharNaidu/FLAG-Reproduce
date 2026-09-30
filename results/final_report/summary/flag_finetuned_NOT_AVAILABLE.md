# flag_finetuned (+FLAG*) - NOT AVAILABLE

No run of this variant exists in results/raw/ or results/flag_md/ at the time this report was built. It requires a GPU-generated 'residual' LLM-text cache (cache/llm/*residual*.json, then its Sentence-BERT encoding in cache/embeddings/) that is not present on this machine: cache/llm/ contains 0 files.

This report does not substitute, estimate or interpolate a number for this cell.

To add it: generate the discriminative AND residual LLM text on a GPU (`python -m scripts.llm.generate_text --kind both`), encode it (`python -m scripts.preprocess.encode_llm_text --kind both`), then run `python -m scripts.train.run --variant flag_finetuned ...` and re-run this script.
