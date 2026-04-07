import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from deep_translator import GoogleTranslator

@st.cache_data(show_spinner=False)
def traduire(texte):
    try:
        return GoogleTranslator(source='auto', target='fr').translate(texte)
    except Exception:
        return texte

# ── Configuration ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500"

# ── Chargement et nettoyage ────────────────────────────────────────────────────
GDRIVE_FILE_ID = "1PoXSkDgs4eH3VvHQk8jEc0faYhHrSZPh"
LOCAL_PATH = "data/raw/TMDB_movie_dataset_v11.csv"

@st.cache_data
def load_data():
    import os, gdown
    if not os.path.exists(LOCAL_PATH):
        os.makedirs("data/raw", exist_ok=True)
        gdown.download(id=GDRIVE_FILE_ID, output=LOCAL_PATH, quiet=False)

    df = pd.read_csv(LOCAL_PATH,
        engine='python',
        on_bad_lines='skip'
    )

    df = df.drop([
        'id', 'status', 'backdrop_path', 'homepage', 'imdb_id',
        'tagline', 'production_companies', 'production_countries',
        'keywords', 'revenue', 'budget', 'original_title', 'spoken_languages'
    ], axis=1)

    df = df.dropna(subset=['title', 'release_date', 'genres'])
    df['genres'] = df['genres'].apply(
        lambda x: [genre.strip() for genre in x.split(',')]
    )
    df['vote_average'] = df['vote_average'].round(1)
    df['year'] = pd.to_datetime(df['release_date']).dt.year

    mask_low = (df['vote_average'] < 5) & (df['vote_count'] >= 3)
    mask_high = (df['vote_average'] >= 5) & (df['vote_count'] >= 10)
    df = df[mask_low | mask_high].copy()

    df = df[
        (df['overview'].notna()) &
        (df['overview'] != '')
    ].reset_index(drop=True)

    return df

# ── Entraînement TF-IDF ────────────────────────────────────────────────────────
@st.cache_resource
def train_model(df):
    df['genres_str'] = df['genres'].apply(lambda x: ' '.join(x))

    vec_genres = TfidfVectorizer(stop_words='english')
    mat_genres = vec_genres.fit_transform(df['genres_str'])

    vec_overview = TfidfVectorizer(stop_words='english', min_df=2)
    mat_overview = vec_overview.fit_transform(df['overview'].fillna(''))

    return mat_genres, mat_overview

# ── Fonction de recommandation hybride ────────────────────────────────────────
def recommend_hybrid(title, df, mat_genres, mat_overview,
                     top_n=5, poids_genre=0.3, note_min=5.0):

    if title not in df['title'].values:
        return None

    poids_overview = 1.0 - poids_genre
    pos = df[df['title'] == title].index[0]
    film_genres = set(df.iloc[pos]['genres'])

    score_genre = cos_sim(mat_genres[pos], mat_genres).flatten()
    score_overview = cos_sim(mat_overview[pos], mat_overview).flatten()
    score_final = (poids_genre * score_genre) + (poids_overview * score_overview)

    top_indices = score_final.argsort()[::-1]

    def clean_title(t):
        return set(re.sub(r'[^\w\s]', '', t.lower()).split())

    title_words = clean_title(title)
    results = []

    for idx in top_indices:
        if idx == pos:
            continue

        film = df.iloc[idx]

        if film['vote_average'] < note_min:
            continue

        film_title_words = clean_title(film['title'])
        words_in_common = title_words & film_title_words
        if len(title_words) > 0 and len(words_in_common) / len(title_words) > 0.5:
            continue

        if 'Documentary' in film['genres'] and 'Documentary' not in film_genres:
            continue

        results.append(idx)

        if len(results) == top_n:
            break

    if not results:
        return None

    recommendations = df.iloc[results].copy()
    recommendations = recommendations.sort_values(
        by='vote_average', ascending=False
    )

    return recommendations[[
        'title', 'genres', 'year', 'vote_average', 'overview', 'poster_path'
    ]]

# ── Chargement des données ─────────────────────────────────────────────────────
try:
    df = load_data()
    mat_genres, mat_overview = train_model(df)
    data_loaded = True
except Exception as e:
    st.error(f"Erreur lors du chargement : {e}")
    data_loaded = False

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🎬 CineMatch")
st.sidebar.markdown("*Ton assistant de recommandation de films*")

menu = st.sidebar.radio(
    "Navigation :",
    [
        "Accueil",
        "Analyse des données",
        "Recommandation",
        "À propos",
    ]
)

if data_loaded:
    st.sidebar.divider()
    st.sidebar.metric("Films disponibles", f"{len(df):,}")
    st.sidebar.image("assets/banner.png", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ACCUEIL
# ══════════════════════════════════════════════════════════════════════════════
if menu == "Accueil":
    st.title("CineMatch")
    st.markdown("### Bienvenue sur ton assistant de recommandation de films !")

    with st.expander("Comment ça marche ?", expanded=True):
        st.markdown("""
        1. **Données** : chargement du dataset TMDB avec plus d'1 million de films
        2. **Nettoyage** : filtrage des films sans votes fiables et sans description
        3. **Modèle** : combinaison de TF-IDF et similarité cosinus
        4. **Recommandation** : système hybride genres + description
        5. **Personnalisation** : choix de l'importance des genres vs description
        """)

    if data_loaded:
        col1, col2, col3 = st.columns(3)
        col1.metric("Films disponibles", f"{len(df):,}")
        col2.metric("Note moyenne", f"{df['vote_average'].mean():.1f}/10")
        col3.metric("Genres uniques",
                    f"{len(set([g for genres in df['genres'] for g in genres]))}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYSE DES DONNÉES
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "Analyse des données":
    st.title("Analyse des données")

    if not data_loaded:
        st.warning("Les données ne sont pas chargées.")
    else:
        tab1, tab2, tab3 = st.tabs([
            "Aperçu du dataset",
            "Statistiques",
            "Visualisations"
        ])

        with tab1:
            st.subheader("Extrait du dataset")
            st.dataframe(
                df[['title', 'genres', 'vote_average',
                    'vote_count', 'year', 'overview']].head(10),
                use_container_width=True
            )

        with tab2:
            st.subheader("Statistiques générales")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Films", f"{len(df):,}")
            col2.metric("Note moyenne", f"{df['vote_average'].mean():.1f}/10")
            col3.metric("Votes moyens", f"{int(df['vote_count'].mean()):,}")
            col4.metric("Durée moyenne", f"{int(df['runtime'].mean())} min")

            st.divider()

            with st.expander("Statistiques détaillées", expanded=False):
                st.dataframe(
                    df[['vote_average', 'vote_count',
                        'popularity', 'runtime']].describe(),
                    use_container_width=True
                )

        with tab3:
            import matplotlib.pyplot as plt
            import seaborn as sns

            st.subheader("Distribution des notes")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(df['vote_average'], bins=20,
                    color='steelblue', edgecolor='white')
            ax.set_title("Distribution des notes après filtrage")
            ax.set_xlabel("Note")
            ax.set_ylabel("Nombre de films")
            st.pyplot(fig)

            st.divider()

            st.subheader("Top 10 des genres les plus fréquents")
            df_genres = df.explode('genres')
            genre_counts = df_genres['genres'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(genre_counts.index, genre_counts.values,
                   color='salmon', edgecolor='white')
            ax.set_title("Top 10 des genres")
            ax.set_xlabel("Genre")
            ax.set_ylabel("Nombre de films")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

            st.divider()

            st.subheader("Note moyenne par genre")
            avg_by_genre = (
                df_genres.groupby('genres')['vote_average']
                .mean()
                .sort_values(ascending=False)
                .head(10)
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(avg_by_genre.index, avg_by_genre.values,
                    color='mediumseagreen', edgecolor='white')
            ax.set_title("Note moyenne par genre (Top 10)")
            ax.set_xlabel("Note moyenne")
            plt.tight_layout()
            st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RECOMMANDATION
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "Recommandation":
    st.image("assets/banniere3.png", use_container_width=True)

    if not data_loaded:
        st.warning("Les données ne sont pas chargées.")
    else:
        # Filtres
        all_genres = sorted(set(g for genres in df['genres'] for g in genres))
        year_min = int(df['year'].min())
        year_max = int(df['year'].max())

        col_fg, col_fa = st.columns(2)
        with col_fg:
            filtre_genres = st.multiselect(
                "Filtrer par genre(s)",
                options=all_genres,
                default=[],
                help="Laisser vide pour ne pas filtrer"
            )
        with col_fa:
            filtre_annees = st.slider(
                "Filtrer par période de sortie",
                min_value=year_min,
                max_value=year_max,
                value=(year_min, year_max),
                step=1
            )

        # Sélection du film filtré
        df_filtre = df[
            (df['year'] >= filtre_annees[0]) &
            (df['year'] <= filtre_annees[1])
        ]
        if filtre_genres:
            df_filtre = df_filtre[
                df_filtre['genres'].apply(
                    lambda g: any(genre in g for genre in filtre_genres)
                )
            ]

        film_list = sorted(df_filtre['title'].tolist())
        if not film_list:
            st.warning("Aucun film ne correspond aux filtres sélectionnés.")
            st.stop()

        film_choisi = st.selectbox(
            f"Choisis un film ({len(film_list):,} disponibles) :",
            film_list
        )
        traduire_fr = st.toggle("Traduire les résumés en français", value=False)

        if film_choisi:
            film_info = df[df['title'] == film_choisi].iloc[0]

            # Infos du film choisi
            st.divider()
            col1, col2 = st.columns([1, 3])

            with col1:
                if (pd.notna(film_info.get('poster_path')) and
                        film_info['poster_path'] != ''):
                    st.image(
                        TMDB_IMAGE_URL + film_info['poster_path'],
                        width=200
                    )
                else:
                    st.markdown("Pas d'affiche disponible")

            with col2:
                st.subheader(f"{film_choisi}")
                overview_film = traduire(film_info['overview']) if traduire_fr else film_info['overview']
                st.markdown(f"**Description :** {overview_film}")
                st.markdown(
                    f"**Genres :** {', '.join(film_info['genres'])}"
                )
                if pd.notna(film_info.get('year')):
                    st.markdown(f"**Année :** {int(film_info['year'])}")

                col_note, col_votes = st.columns(2)
                with col_note:
                    st.metric("Note",
                              f"{film_info['vote_average']}/10")
                with col_votes:
                    st.metric("Votes",
                              f"{int(film_info['vote_count']):,}")

            st.divider()

            # Paramètres
            with st.container(border=True):
                st.subheader("Paramètres de recommandation")

                poids_genre = st.slider(
                    "Importance des genres vs description",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.1,
                    help="0.0 = description uniquement | 1.0 = genres uniquement"
                )

                poids_overview = round(1.0 - poids_genre, 1)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Poids genres",
                              f"{int(poids_genre * 100)}%")
                with col2:
                    st.metric("Poids description",
                              f"{int(poids_overview * 100)}%")

                if poids_genre == 0.0:
                    st.info(
                        "Recommandation basée uniquement sur la description"
                    )
                elif poids_genre <= 0.3:
                    st.info(
                        "La description a plus d'importance que les genres"
                    )
                elif poids_genre == 0.5:
                    st.info(
                        "Équilibre parfait entre genres et description"
                    )
                elif poids_genre <= 0.7:
                    st.info(
                        "Les genres ont plus d'importance que la description"
                    )
                else:
                    st.info(
                        "Recommandation basée principalement sur les genres"
                    )


            st.divider()

            # Bouton recommandation
            if st.button(
                "Trouver des films similaires",
                type="primary",
                use_container_width=True
            ):
                with st.spinner("Recherche en cours..."):
                    resultats = recommend_hybrid(
                        film_choisi, df, mat_genres, mat_overview,
                        top_n=5,
                        poids_genre=poids_genre,
                        note_min=5.0
                    )

                if resultats is None:
                    st.warning("Aucun film similaire trouvé.")
                else:
                    # Appliquer les filtres
                    if filtre_genres:
                        resultats = resultats[
                            resultats['genres'].apply(
                                lambda g: any(genre in g for genre in filtre_genres)
                            )
                        ]
                    resultats = resultats[
                        (resultats['year'] >= filtre_annees[0]) &
                        (resultats['year'] <= filtre_annees[1])
                    ]

                    if resultats.empty:
                        st.warning(
                            "Aucun film similaire trouvé avec ces filtres."
                        )
                    else:
                        st.subheader(
                            f"Films similaires à **{film_choisi}** :"
                        )

                        for _, row in resultats.iterrows():
                            col1, col2 = st.columns([1, 3])

                            with col1:
                                if (pd.notna(row.get('poster_path')) and
                                        row['poster_path'] != ''):
                                    st.image(
                                        TMDB_IMAGE_URL + row['poster_path'],
                                        width=150
                                    )
                                else:
                                    st.markdown("Pas d'affiche")

                            with col2:
                                st.markdown(f"### {row['title']}")
                                col_note, col_annee = st.columns(2)
                                with col_note:
                                    st.metric(
                                        "Note",
                                        f"{row['vote_average']}/10"
                                    )
                                with col_annee:
                                    if pd.notna(row.get('year')):
                                        st.metric(
                                            "Année",
                                            int(row['year'])
                                        )
                                st.markdown(
                                    f"**Genres :** {', '.join(row['genres'])}"
                                )
                                overview_rec = traduire(row['overview']) if traduire_fr else row['overview']
                                st.markdown(
                                    f"**Description :** {overview_rec}"
                                )

                            st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — À PROPOS
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "À propos":
    st.title("À propos du projet")

    st.markdown("""
    ### Présentation du projet

    Ce projet est un système de recommandation de films basé sur le dataset TMDB
    (The Movie Database) contenant plus d'1 million de films.

    ### Méthodologie

    **Nettoyage des données :**
    - Suppression des colonnes inutiles
    - Filtrage intelligent selon le nombre de votes
    - Conservation uniquement des films avec une description

    **Système de recommandation :**
    - **TF-IDF** : transforme le texte en vecteurs numériques
    - **Similarité cosinus** : mesure la proximité entre les films
    - **Système hybride** : combine score genres et score description

    **Optimisations réalisées :**
    - Pondération dégressive des genres (poids optimal = 3)
    - Comparaison sublinear_tf vs normal
    - Filtre sur la note minimum des recommandations
    - Exclusion des titres trop similaires et des doublons

    ### Technologies utilisées
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Données**")
        st.markdown("- Pandas\n- NumPy")
    with col2:
        st.markdown("**Machine Learning**")
        st.markdown("- Scikit-learn\n- TF-IDF\n- KNN")
    with col3:
        st.markdown("**Application**")
        st.markdown("- Streamlit\n- TMDB API")

    st.divider()
    st.markdown("""
    ### Pistes d'amélioration
    """)
    st.checkbox("Ajouter un filtre par genre dans la recherche", value=True, disabled=True)
    st.checkbox("Ajouter un filtre par année de sortie", value=True, disabled=True)
    st.checkbox("Intégrer les notes des utilisateurs (collaborative filtering)")
    st.checkbox("Améliorer la qualité des recommandations avec plus de features")

st.sidebar.divider()
st.caption("CineMatch — Recommandation de films — 2025")