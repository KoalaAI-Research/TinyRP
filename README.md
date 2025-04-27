# TinyRP
 TinyRP model training code

## How to use:
Run the following scripts in order:
* dataset_get_aesir.py
* train_tokenizer.py
* eval_tokenizer.py
* tokenize_folder.py (tokenizes the aesir dataset)
* dataset_get_pippa.py (needs to have the tokenizer ready)
* merge_bins.py (merges the train .bin files)
* train_model.py
