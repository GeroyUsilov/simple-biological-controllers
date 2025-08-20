# Simple biological controllers: code & analysis

Code and notebooks accompanying  
**Russo, Husain, Ranganathan, Pincus, Murugan (2025)** — *Simple biological controllers drive the evolution of soft modes* ([arXiv:2507.11973](https://arxiv.org/abs/2507.11973)).

## Repo layout
- `mace_et_al_analysis/` — analyses using Mace et al. (2020) data  
- `costanzo_et_al_analysis/` — analyses using Costanzo et al. (2016, 2021) data  
- `mm_simulations/` — minimal model simulations for the manuscript

## Requirements
Python 3.9+ with Jupyter. Typical deps: `numpy`, `pandas`, `scipy`, `matplotlib`.

## Data (must be downloaded separately)
Place the raw/processed files from the papers’ data deposits into a local `data/` folder, e.g.:
mace_et_al_analysis/data/ # transcriptomics (see paper’s Data Availability)
costanzo_et_al_analysis/data/ # global genetic interaction network and environment-specific data

Sources:
- Mace et al., PLOS ONE 2020 — “Multi-kinase control of environmental stress responsive transcription” (GEO: GSE115556).  
  https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0230246
- Costanzo et al., Science 2016 — “A global genetic interaction network maps a wiring diagram of cellular function.”  
  https://pubmed.ncbi.nlm.nih.gov/27708008/
- Costanzo et al., Science 2021 — “Environmental robustness of the global yeast genetic interaction network.”  
  https://pubmed.ncbi.nlm.nih.gov/33958448/


## Quick start
1) Create `data/` as above and copy the downloaded files.  
2) Open the notebooks in each folder and run all cells.

## Citation
If you use this code, please cite the arXiv paper above.

## License
MIT (see `LICENSE`).