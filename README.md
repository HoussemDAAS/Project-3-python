# Project-3-python

A Python project with Jupyter notebook starter guide.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook

### Installation

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Starter Notebook

To get started with this project, open the Jupyter starter notebook:

```bash
jupyter notebook starter.ipynb
```

This interactive notebook will guide you through:
- Basic Python concepts
- Project structure
- Example code and exercises

## Étapes du Projet

### Étape 1 — Collecte & Stockage
- Importer/collecter les datasets publics:
  - Fake News Detection Dataset (Kaggle)
  - Hate Speech and Offensive Language Dataset
  - Arabic Offensive Tweets Dataset
- Nettoyage:
  - Suppression des URLs, symboles, emojis
  - Conversion en minuscules
  - Tokenization et lemmatisation (NLTK ou spaCy)
  - Traduction vers une langue pivot (anglais ou français) si nécessaire
- Stockage SQL:
  - Base MySQL `dataControl`, tables `news` et `labeled`
  - Réalisé via `starter.ipynb` (mode `process`), puis réexécution en mode `load`

### Étape 2 — Analyse
- Chargement depuis MySQL (fallback CSV si la connexion échoue)
- Analyses générées:
  - Distribution des classes (fake/hate/normal)
  - Word clouds
  - Statistiques lexicales: longueur moyenne, top mots
  - Détection de mots sensibles récurrents (FR + EN)
- Notebook: `analysis_step_2.ipynb`
- Script alternatif (exécutable depuis le starter): `%run analysis_step_2.py`

### Étape 3 — Modélisation (Deep Learning)

#### Recommandation
- Utiliser un **transformer fine‑tuned**, pas un LLM générique.
- Raisons:
  - Les LLM (APIs) sont coûteux et difficiles à évaluer de manière reproductible.
  - Les petits transformers fine‑tunés (DistilBERT) offrent une excellente F1/accuracy avec peu de ressources et une évaluation propre hors‑ligne.
  - Si vous gardez des données non‑anglophones sans traduction, utilisez **XLM‑Roberta base** (multilingue).

#### Modèles
- Baseline: **TF‑IDF + Logistic Regression**
  - Référence rapide pour vérifier la séparabilité et l’étiquetage.
- **DistilBERT (recommandé)**
  - Fine‑tune par dataset:
    - NEWS: `fake` vs `normal` (depuis la colonne `label`).
    - LABELED: `hate` (classes 0/1) vs `normal` (classe 2).
  - Fonctionne sur CPU; gardez `epochs=2–3`, `batch_size=8–16`, `max_length=256`.
- **Multilingue**
  - Si vous conservez arabe/français sans traduction: `xlm-roberta-base`.
  - Sinon, le pipeline de traduction vers l’anglais + DistilBERT est idéal.

#### Évaluation
- Par modèle: Accuracy, Macro F1, classification report, matrice de confusion.
- Comparez baseline vs DistilBERT; sélectionnez le meilleur par F1 validation.
- Gérez le déséquilibre: splits stratifiés; `class_weight` pour LogReg ou oversampling.

#### Notebook & Exécution
- Notebook ajouté: `modeling_step_3.ipynb`
  - Connexion MySQL (`dataControl`) et chargement de `news` et `labeled` (fallback CSV).
  - Exécute la baseline TF‑IDF + LR et le fine‑tuning DistilBERT.
  - Sauvegarde des modèles: `models/news_distilbert` et `models/labeled_distilbert`.

#### Intégration dans le Starter
- Option A (recommandée): exécuter les notebooks dédiés.
  - Ouvrez `analysis_step_2.ipynb` puis `modeling_step_3.ipynb` avec le kernel `Project3 venv`.
- Option B: appeler depuis `starter.ipynb` en fin de pipeline:
  - Ajoutez une cellule:
    ```python
    # Étape 2 — Analyses
    %run analysis_step_2.py

    # Étape 3 — Modélisation
    %run modeling_step_3.ipynb
    ```

## Environnements & Dépendances

### Kernel Jupyter
- Sélectionnez le kernel `Project3 venv` (enregistré via `ipykernel`).

### Installation
- Installez les paquets d’analyse et de training dans votre venv:
  - `./venv/Scripts/python.exe -m pip install matplotlib seaborn wordcloud pymysql`
  - `./venv/Scripts/python.exe -m pip install transformers scikit-learn`
- Torch (CPU) si nécessaire:
  - `./venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu`

## Features

- Interactive Jupyter notebook for learning
- Basic Python examples
- Easy setup and configuration

## Contributing

Feel free to contribute to this project by opening issues or submitting pull requests.

## License

This project is open source.