import ast
import streamlit as st
import pandas as pd
import re
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from deep_translator import GoogleTranslator

@st.cache_data(show_spinner=True)
# Traduit un texte en français via Google Translate.
# Retourne le texte original en cas d'erreur.
def traduire(texte):
    try:
        return GoogleTranslator(source='auto', target='fr').translate(texte)
    except Exception:
        return texte

# Configuration
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500"

def afficher_affiche(film, width=150):
    if pd.notna(film.get('poster_path')) and film['poster_path'] != '':
        st.image(TMDB_IMAGE_URL + film['poster_path'], width=width)
    else:
        st.markdown("Pas d'affiche disponible")

# Chargement et nettoyage
# Charge le dataset brut et le dataset nettoyé.
# Retourne (df_raw, df) — df_raw pour les graphes avant filtrage, df pour les recommandations.
@st.cache_data
def load_data():
    df_raw = pd.read_csv("data/raw/TMDB_movie_dataset_v11.csv",
        engine='python',
        on_bad_lines='skip'
    )
    df = pd.read_csv("data/processed/TMDB_cleaned.csv")
    df['genres'] = df['genres'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    return df_raw, df

# Entraînement TF-IDF
# Calcule les matrices TF-IDF des genres et des descriptions.
# Mis en cache avec cache_resource pour n'être exécuté qu'une seule fois au démarrage.
@st.cache_resource
def train_model(df):
   
    df['genres_str'] = df['genres'].apply(lambda x: ' '.join(x))

    vec_genres = TfidfVectorizer(stop_words='english')
    mat_genres = vec_genres.fit_transform(df['genres_str'])

    vec_overview = TfidfVectorizer(stop_words='english', min_df=2)
    mat_overview = vec_overview.fit_transform(df['overview'].fillna(''))

    return mat_genres, mat_overview

# Fonction de recommandation hybride
# Retourne les top 5 films les plus similaires au titre donné.
# Combine similarité cosinus sur les genres et les descriptions selon poids_genre.
# Exclut les films trop similaires dans le titre, les documentaires hors contexte,
# et les films sous la note minimale.
def recommend_hybrid(title, df, mat_genres, mat_overview,
                     top_n=5, poids_genre=0.5, note_min=5.0):
    

    # Vérifier que le film existe dans le dataset
    if title not in df['title'].values:
        return None

    # Calculer le poids de la description (complément du poids genre)
    poids_overview = 1.0 - poids_genre
    # Récupérer la position du film dans le DataFrame
    pos = df[df['title'] == title].index[0]
    # Récupérer les genres du film pour filtrer les documentaires plus tard
    film_genres = set(df.iloc[pos]['genres'])

    # Calculer la similarité cosinus entre le film et tous les autres sur les genres
    score_genre = cos_sim(mat_genres[pos], mat_genres).flatten()
    # Calculer la similarité cosinus entre le film et tous les autres sur les descriptions
    score_overview = cos_sim(mat_overview[pos], mat_overview).flatten()
    # Combiner les deux scores selon les poids choisis par l'utilisateur
    score_final = (poids_genre * score_genre) + (poids_overview * score_overview)

    # Trier les indices par score décroissant
    top_indices = score_final.argsort()[::-1]

    # Convertit un titre en ensemble de mots (minuscules, sans ponctuation)
    # pour pouvoir comparer les titres et éviter les recommandations trop évidentes
    def clean_title(t):
        return set(re.sub(r'[^\w\s]', '', t.lower()).split())

    title_words = clean_title(title)
    results = []

    for idx in top_indices:
        # Ignorer le film lui-même
        if idx == pos:
            continue

        film = df.iloc[idx]

        # Ignorer les films sous la note minimale
        if film['vote_average'] < note_min:
            continue

        # Ignorer les films dont le titre partage plus de 50% des mots avec le film choisi
        film_title_words = clean_title(film['title'])
        words_in_common = title_words & film_title_words
        if len(title_words) > 0 and len(words_in_common) / len(title_words) > 0.5:
            continue

        # Ignorer les documentaires si le film choisi n'en est pas un
        if 'Documentary' in film['genres'] and 'Documentary' not in film_genres:
            continue

        results.append(idx)

        # Arrêter quand on a assez de résultats
        if len(results) == top_n:
            break

    if not results:
        return None

    # Trier les résultats par note décroissante
    recommendations = df.iloc[results].copy()
    recommendations = recommendations.sort_values(
        by='vote_average', ascending=False
    )

    return recommendations[[
        'title', 'genres', 'year', 'vote_average', 'overview', 'poster_path'
    ]]

# Chargement des données
try:
    df_raw, df = load_data()
    mat_genres, mat_overview = train_model(df)
    data_loaded = True
except Exception as e:
    st.error(f"Erreur lors du chargement : {e}")
    data_loaded = False

# Sidebar
st.sidebar.title("🎬 CineMatch")
st.sidebar.markdown("*Ton assistant de recommandation de films*")

menu = st.sidebar.radio(
    "Navigation :",
    [
        "Accueil",
        "Analyse des données",
        "Cinématch",
        "À propos",
    ]
)

if data_loaded:
    st.sidebar.divider()
    st.sidebar.metric("Films disponibles", f"{len(df):,}")
    st.sidebar.image("assets/banner.png", use_container_width=True)

# PAGE 1 — ACCUEIL

if menu == "Accueil":
    st.title("CineMatch")
    st.markdown("### Bienvenue sur ton assistant de recommandation de films !")

    with st.expander("Comment ça marche ?", expanded=True):
        st.markdown("""
        1. :material/database: **Données** : chargement du dataset TMDB avec plus d'1 million de films
        2. :material/cleaning_services: **Nettoyage** : filtrage des films sans votes fiables et sans description
        3. :material/model_training: **Modèle** : combinaison de TF-IDF et similarité cosinus
        4. :material/recommend: **Recommandation** : système hybride genres + description
        5. :material/tune: **Personnalisation** : choix de l'importance des genres vs description
        """)

    if data_loaded:
        col1, col2, col3 = st.columns(3)
        col1.metric("Films disponibles", f"{len(df):,}")
        col2.metric("Note moyenne", f"{df['vote_average'].mean():.1f}/10")
        col3.metric("Genres uniques",
                    f"{len(set([g for genres in df['genres'] for g in genres]))}")

# PAGE 2 — ANALYSE DES DONNÉES

elif menu == "Analyse des données":
    st.title("Analyse des données")
    st.caption("Compare les données brutes et les données nettoyées avant de lancer une recommandation.")

    if not data_loaded:
        st.warning("Les données ne sont pas chargées.")
    else:
        tab1, tab2, tab3, tab4 = st.tabs([
            "Aperçu brut",
            "Aperçu nettoyé",
            "Statistiques",
            "Visualisations"
        ])

        with tab1:
            st.subheader("Extrait du dataset brut")
            st.dataframe(df_raw.head(10), use_container_width=True)

        with tab2:
            st.subheader("Extrait du dataset nettoyé")
            st.dataframe(df.head(10), use_container_width=True)

        with tab3:
            st.subheader("Statistiques générales")
            m1, m2 = st.columns(2)
            m1.metric("Lignes brutes", f"{len(df_raw):,}")
            m2.metric("Lignes nettoyées", f"{len(df):,}")
            st.divider()

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

        with tab4:
            st.subheader("Distribution des notes avant filtrage")
            st.caption("La majorité des films ont une note de 0 car ils n'ont reçu aucun vote. Le dataset brut est très bruité.")
            fig = px.histogram(df_raw, x='vote_average', nbins=20, color_discrete_sequence=['steelblue'])
            fig.update_layout(xaxis_title="Note", yaxis_title="Nombre de films")
            st.plotly_chart(fig, use_container_width=False)

            st.divider()

            st.subheader("Distribution des notes (>= 1 vote)")
            st.caption("En ne gardant que les films avec au moins 1 vote, le pic à 0 disparaît mais la distribution reste déséquilibrée vers les notes basses.")
            df_vote1 = df_raw[df_raw['vote_count'] >= 1]
            fig = px.histogram(df_vote1, x='vote_average', nbins=20, color_discrete_sequence=['steelblue'])
            fig.update_layout(xaxis_title="Note", yaxis_title="Nombre de films")
            st.plotly_chart(fig, use_container_width=False)

            st.divider()

            st.subheader("Distribution des notes après filtrage")
            st.caption("Après filtrage (≥ 3 votes si note < 5, ≥ 10 votes sinon), la distribution est plus fiable et centrée autour de 6-7.")
            fig = px.histogram(df, x='vote_average', nbins=20, color_discrete_sequence=['salmon'])
            fig.update_layout(xaxis_title="Note", yaxis_title="Nombre de films")
            st.plotly_chart(fig, use_container_width=False)

            st.divider()

            st.subheader("Top 10 des genres les plus fréquents")
            st.caption("Le Drama et la Comédie dominent largement le dataset, reflétant la production mondiale de films.")
            df_genres = df.explode('genres')
            genre_counts = df_genres['genres'].value_counts().head(10).reset_index()
            genre_counts.columns = ['genres', 'count']
            fig = px.bar(genre_counts, x='genres', y='count', color_discrete_sequence=['steelblue'])
            fig.update_layout(xaxis_title="Genre", yaxis_title="Nombre de films")
            st.plotly_chart(fig, use_container_width=False)

            st.divider()

            st.subheader("Nombre de films par année")
            st.caption("La production cinématographique mondiale explose à partir des années 2000 grâce à la démocratisation des outils de tournage et de distribution.")
            films_per_year = df['year'].value_counts().sort_index().reset_index()
            films_per_year.columns = ['year', 'count']
            fig = px.line(films_per_year, x='year', y='count', color_discrete_sequence=['steelblue'])
            fig.update_layout(xaxis_title="Année", yaxis_title="Nombre de films")
            st.plotly_chart(fig, use_container_width=False)

            st.divider()

            st.subheader("Note moyenne par année")
            st.caption("Les films anciens ont tendance à avoir de meilleures notes : seuls les classiques ont survécu au temps et sont encore notés aujourd'hui.")
            avg_per_year = df.groupby('year')['vote_average'].mean().reset_index()
            fig = px.line(avg_per_year, x='year', y='vote_average', color_discrete_sequence=['salmon'])
            fig.update_layout(xaxis_title="Année", yaxis_title="Note moyenne")
            st.plotly_chart(fig, use_container_width=False)

            st.divider()

            st.subheader("Matrice de corrélation")
            st.caption("La popularité est faiblement corrélée à la note — un film peut être très populaire sans être bien noté, et vice versa.")
            corr = df[['vote_average', 'vote_count', 'popularity', 'runtime']].corr()
            fig = ff.create_annotated_heatmap(
                z=corr.values.round(2),
                x=list(corr.columns),
                y=list(corr.index),
                colorscale='RdBu',
                showscale=True
            )
            st.plotly_chart(fig, use_container_width=False)

# PAGE 3 — RECOMMANDATION

elif menu == "Cinématch":
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
                afficher_affiche(film_info, width=200)

            with col2:
                overview_film = traduire(film_info['overview']) if traduire_fr else film_info['overview']
                st.markdown(":material/description: **Description :**")
                st.markdown(overview_film)
                st.markdown(
                    f":material/theaters: **Genres :** {', '.join(film_info['genres'])}"
                )
                if pd.notna(film_info.get('year')):
                    st.markdown(f":material/calendar_today: **Année :** {int(film_info['year'])}")

                col_note, col_votes = st.columns(2)
                with col_note:
                    st.metric(":material/star: Note",
                              f"{film_info['vote_average']}/10")
                with col_votes:
                    st.metric(":material/group: Votes",
                              f"{int(film_info['vote_count']):,}")

            # Paramètres de recommandation formulaire
            with st.form("recommandation"):
                st.subheader("Paramètres de recommandation")

                choix = st.select_slider(
                    "Choisis ta préférence de recommandation :",
                    options=["Description", "Mixte", "Genres"],
                    value="Mixte"
                )
                poids_map = {"Description": 0.0, "Mixte": 0.5, "Genres": 1.0}
                poids_genre = poids_map[choix]

                submitted = st.form_submit_button(
                    "Trouver des films similaires",
                    type="primary",
                    use_container_width=True
                )

            if submitted:
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
                                afficher_affiche(row, width=150)

                            with col2:
                                st.markdown(f"### {row['title']}")
                                col_note, col_annee = st.columns(2)
                                with col_note:
                                    st.metric(
                                        ":material/star: Note",
                                        f"{row['vote_average']}/10"
                                    )
                                with col_annee:
                                    if pd.notna(row.get('year')):
                                        st.metric(
                                            ":material/calendar_today: Année",
                                            int(row['year'])
                                        )
                                st.markdown(
                                    f":material/theaters: **Genres :** {', '.join(row['genres'])}"
                                )
                                overview_rec = traduire(row['overview']) if traduire_fr else row['overview']
                                st.markdown(
                                    f":material/description: **Description :** {overview_rec}"
                                )

                            st.divider()

# PAGE 4 — À PROPOS

elif menu == "À propos":
    st.title("À propos du projet")

    st.markdown("""
    ### Présentation du projet

    CineMatch est un système de recommandation de films basé sur le dataset TMDB
    (The Movie Database) contenant plus d'1 million de films.
    L'objectif est de proposer des films similaires à partir d'un titre choisi,
    en combinant les genres et les descriptions.
    """)

    st.divider()

    st.markdown("### Méthodologie")

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=False):
            st.markdown("**Nettoyage des données**")
            st.markdown("""
            - Suppression des colonnes inutiles
            - Filtrage selon le nombre de votes
            - Suppresssion des films avec note 0
            - Conservation des films avec description
            - Export du dataset nettoyé dans `data/processed/`
            """)
    with col2:
        with st.container(border=False):
            st.markdown("**Système de recommandation**")
            st.markdown("""
            - TF-IDF: transforme le texte en vecteurs numériques
            - Similarité cosinus: mesure la proximité entre films
            - Système hybride: combine genres et description
            - Filtre sur la note minimale des recommandations
            - Exclusion des titres trop similaires et des documentaires
            """)

    st.divider()

    st.markdown("### Technologies utilisées")
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=False):
            st.markdown("**Données**")
            st.markdown("""
            - Pandas 
            - NumPy
            - TMDB Dataset
            """)
    with col2:
        with st.container(border=False):
            st.markdown("**Similarité**")
            st.markdown("""
                        - Scikit-learn
                        - TF-IDF
                        - Similarité cosinus
                        """)
    with col3:
        with st.container(border=False):
            st.markdown("**Application**")
            st.markdown("""
            - Streamlit
            - Google Colab
            - deep-translator
            """)


st.caption("CineMatch — Recommandation de films — 2026")