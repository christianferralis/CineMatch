# CineMatch — Système de recommandation de films

Application Streamlit de recommandation de films basée sur le dataset TMDB (The Movie Database).

---

## Structure du projet

```
cinematch/
├── app.py                        # Application Streamlit principale
├── requirements.txt              # Dépendances Python
├── data/
│   ├── raw/TMDB_movie_dataset_v11.csv     # Dataset brut (1M+ films)
│   └── processed/TMDB_cleaned.csv        # Dataset nettoyé et exporté depuis le notebook
├── notebooks/Movies.ipynb        # Exploration, nettoyage et développement du système
└── assets/                       # Images et bannières de l'app
```

---

## Lancer l'application

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Les données

### Pourquoi deux datasets ?
- `df_raw` : le dataset brut, utilisé uniquement pour afficher les graphes "avant filtrage"
- `df` : le dataset nettoyé, utilisé pour les recommandations

### Pourquoi garder les films mal notés (`mask_low`) ?
Les films notés < 5 avec au moins 3 votes sont conservés pour que l'utilisateur puisse les **sélectionner** comme film de départ. Ils ne peuvent pas apparaître dans les **recommandations** (filtrés par `note_min=5.0`).

```python
mask_low = (df['vote_average'] < 5) & (df['vote_count'] >= 3)
mask_high = (df['vote_average'] >= 5) & (df['vote_count'] >= 10)
```

---

## Le modèle — TF-IDF + Similarité Cosinus

### C'est quoi TF-IDF ?
TF-IDF transforme du texte en vecteurs numériques. Chaque mot reçoit un score :
- **TF (Term Frequency)** : fréquence du mot dans le document
- **IDF (Inverse Document Frequency)** : pénalise les mots trop communs dans tous les documents

Résultat : les mots rares et spécifiques ont un poids fort, les mots banals un poids faible.

### Pourquoi `stop_words='english'` ?
Pour ignorer les mots sans sens comme "the", "a", "is" qui pollueraient la comparaison.

### Pourquoi `min_df=2` sur les descriptions ?
Un mot qui n'apparaît que dans un seul film ne permet pas de faire des liens entre films. `min_df=2` ignore les mots présents dans moins de 2 films.

### Pourquoi deux vectorizers séparés (genres et description) ?
Pour pouvoir les combiner avec des poids différents selon la préférence de l'utilisateur.

---

## La similarité cosinus

### Pourquoi cosinus et pas distance euclidienne ?
La similarité cosinus mesure l'**angle** entre deux vecteurs, pas leur distance. Elle est insensible à la longueur des textes — un film avec une longue description ne sera pas favorisé par rapport à un film avec une description courte.

---

## Le système hybride

### Comment ça fonctionne ?
```python
score_final = (poids_genre × score_genres) + (poids_overview × score_description)
```

### Pourquoi combiner genres ET description ?
- **100% genres** : recommande des films du même genre mais sans rapport dans l'histoire
- **100% description** : peut recommander des films de genres différents mais thématiquement proches
- **Hybride** : meilleur équilibre entre cohérence de genre et proximité narrative

---

## Les filtres de `recommend_hybrid`

### Pourquoi `note_min=5.0` ?
Pour ne pas recommander des films de mauvaise qualité. Un utilisateur qui cherche des films similaires s'attend à des films corrects.

### Pourquoi exclure les documentaires ?
Si l'utilisateur cherche un film de fiction, recevoir des documentaires comme recommandation est incohérent. Le filtre les exclut sauf si le film de départ est lui-même un documentaire.

### Pourquoi le filtre sur les titres similaires ?
Pour éviter les recommandations trop évidentes. Si plus de 50% des mots du titre se retrouvent dans un autre titre, ce film est exclu. Exemple : chercher "Avengers" ne retourne pas tous les films "Avengers: ...".

---

## Technologies utilisées

| Catégorie | Librairie |
|---|---|
| Application | Streamlit |
| Données | Pandas, NumPy |
| Similarité | Scikit-learn (TF-IDF, Cosinus) |
| Visualisations | Plotly |
| Traduction | deep-translator (Google Translate) |
| Images | TMDB API (poster_path) |
