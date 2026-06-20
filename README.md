# s1000-diagnoser


In this ongoing project I'm building a proof of concept for audio-based engine fault detection on BMW S1000R Motorbikes. The project is presented in the forum https://www.s1000-forum.de/viewtopic.php?f=5&t=24477 

The most promising strategy at the moment is training a CAE on "healthy" engines, use it to compress and reconstruct MEL-spectral sequences of audiofiles, and compare them to the input spectrum. High deviation from the original indicates that the CAE was not able to properly reconstruct, and thus it must be an audiosequence with an unknown (un-healthy) noise profile.

The work is still ongoing, but needs more data and more time investment to build a more robust pipeline.

## Structure

```text
s1000-diagnoser/
├── data/
│   ├── raw/                # Unveränderte Original-Audiofiles (wav, m4a, mp4)
│   ├── processed/          # Extrahierte Audiospuren, bereinigte Signale
│   └── features/           # Gespeicherte Feature-Vektoren (z.B. .npy oder .pkl)
├── notebooks/
│   ├── 01_manual_audio_exploration.ipynb   # first look at the audio files
│   ├── 02_preprocessing.ipynb              # preprocessing of mel spectrums
│   ├── 04_CAE_Annomalydetector.ipynb       # Notebook to train a simple CAE for Annomaly Detection
│   └── 0X_XXX                              # other notebooks for brainstorming methods
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
