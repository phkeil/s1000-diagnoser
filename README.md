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

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
