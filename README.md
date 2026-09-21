# LSF-SR

Official implementation of **"LSF-SR: Latent Semantic Fusion for Sequential Recommendation via Flow-based Conditional Variational Autoencoders."**

## Environment

Create and activate a Conda environment:

```bash
conda create -n lsf-sr python=3.8.19 -y
conda activate lsf-sr
```

Install PyTorch and the required dependencies:

```bash
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch-scatter==2.1.0+pt113cu117 \
  -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install torch-geometric==2.6.0
pip install -r requirements.txt
```

## Running LSF-SR

The reproduction pipeline consists of three steps.

### 1. Build the datasets

Open the notebook corresponding to the target dataset in `build_datasets_and_prompts/`. Follow the instructions in the notebook to download and preprocess the raw data and generate the required intermediate files.

For example, to prepare the Amazon Beauty dataset, run:

```bash
jupyter notebook build_datasets_and_prompts/Amazon_Beauty.ipynb
```

### 2. Generate item descriptions and semantic embeddings

Use the LLM generation script to obtain textual descriptions of the items, and then convert the generated descriptions into text embedding vectors:

```bash
cd get_llmResponse_and_semanticEmb
python obtain_response_item.py
python obtain_text_emb_item.py
```

Before running these scripts, configure the target dataset paths and the required model credentials in accordance with the comments in the scripts.

In our experiments, we use **`meta-llama/Llama-3.1-8B-Instruct`** for item description generation and **`princeton-nlp/sup-simcse-roberta-large`** for semantic embedding extraction. The models and prompting template can be replaced without changing the downstream pipeline, although the resulting descriptions may affect recommendation performance.

For reproducibility, we provide the complete Amazon Beauty semantic embeddings (`beauty_item_semantic_embeddings.pt`) and the LLM responses for the first 20 items (`response_item_summary_sample.json`). The sample responses help verify the generation procedure but are insufficient to reconstruct the complete embeddings, which are provided separately for direct use in downstream model training.

### 3. Train and evaluate LSF-SR

Run the script corresponding to the target dataset. For example, to reproduce the experiments on Amazon Beauty:

```bash
cd ../recommender_code/scripts
bash beauty.sh
```

The scripts for the other supported datasets are available in the same directory:

```text
office.sh
sports.sh
toys.sh
yelp.sh
```

## Citation

If you find this repository helpful for your work, please cite the following paper:

```bibtex
@inproceedings{chen2026lsfsr,
  title     = {LSF-SR: Latent Semantic Fusion for Sequential Recommendation via Flow-based Conditional Variational Autoencoders},
  author    = {Shih-Hong Chen and Josh Jia-Ching Ying and Vincent S. Tseng},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
  year      = {2026}
}
```
