## decomposing-activations-local-geometry

This is the official repository for **From Directions to Regions: Decomposing Activations in Language Models via Local Geometry** (Or Shafran, Shaked Ronen, Omri Fahn, Shauli Ravfogel, Atticus Geiger, Mor Geva), 2026.

Currently the repo contains:
- An end-to-end MFA tutorial for training, interpretation, visualization, and steering
- A lightweight guide for loading pretrained MFA checkpoints
- Code for FSDP-based multi-GPU training and saving/loading.

### Pretrained MFAs

We include a dedicated notebook with the loading flow for pretrained checkpoints, including download from Hugging Face and instantiating the MFA.

Currently available pretrained checkpoints are the **8k MFAs**:
- **Gemma-2-2B**: layers 6 and 18
- **Llama-3.1-8B**: layers 8 and 22

MFA weights can also be accessed directly at [Hugging Face](https://huggingface.co/ordavids1/decomposing-local-geometry-MFAs).

### More coming soon

We will continue releasing additional pretrained MFAs and broader model/layer coverage, along with more code for reproducing paper experiments.

For questions, feel free to reach out.
