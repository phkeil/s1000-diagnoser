# s1000-diagnoser

Ein kompaktes Projektgerüst zur Diagnose von S1000-Audiodaten.

## Struktur

```text
s1000-diagnoser/
├── data/
│   ├── raw/                # Unveränderte Original-Audiofiles (wav, m4a, mp4)
│   ├── processed/          # Extrahierte Audiospuren, bereinigte Signale
│   └── features/           # Gespeicherte Feature-Vektoren (z.B. .npy oder .pkl)
├── notebooks/
│   ├── 01_eda_audio.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_testing.ipynb
├── src/
│   ├── __init__.py
│   ├── audio_utils.py
│   ├── feature_extraction.py
│   └── model.py
├── reports/
│   └── figures/
├── config/
│   └── config.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

This project now provides a Conda environment for reproducible installs. Two options are shown below.

### Recommended: create a Conda environment

```bash
# create the environment from environment.yml
conda env create -f environment.yml

# activate it
conda activate s1000-diagnoser
```

This uses the `conda-forge` channel and installs the pinned packages. The file `environment.yml` also keeps a pip fallback.
