import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv("books_clean.csv")  

def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text
    return ""


df["title_clean"] = df["title"].astype(str).apply(clean_text)
df["authors"] = df["authors"].astype(str).apply(clean_text)
df["publisher"] = df["publisher"].astype(str).apply(clean_text)
df["language_code"] = df["language_code"].astype(str).apply(clean_text)
df["year"] = df["year"].astype(str).apply(clean_text)


df["bag_of_words"] = (
    df["title_clean"] + " " +
    df["authors"] + " " +
    df["publisher"] + " " +
    df["language_code"] + " " +
    df["year"]
)


tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["bag_of_words"])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index=df["title_clean"]).drop_duplicates()


def recommend_books(title, n=5):
    title = clean_text(title)

    if title not in indices:
        return None

    idx = indices[title]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    similar_indices = [i[0] for i in similarity_scores[1:n+1]]

    return df.iloc[similar_indices][["title", "authors", "publisher", "average_rating"]]

st.title("📚 Book Recommender App")
st.write("Entrez un titre ou un mot clé pour trouver des livres similaires.")


# Champ de recherche
search_query = st.text_input("🔍 Recherche de livres")

if search_query:

    search_clean = clean_text(search_query)

    # Filtrer les livres contenant le mot clé
    results = df[df["title_clean"].str.contains(search_clean)]

    if results.empty:
        st.warning("Aucun livre trouvé pour cette recherche.")
    else:
        st.subheader("📘 Résultats trouvés :")
        for idx, row in results.iterrows():
            st.markdown(f"### {row['title']}")
            st.write(f"Auteur : {row['authors']}")
            st.write(f"Publisher : {row['publisher']}")
            st.write(f"Rating : {row['average_rating']}")

            # Bouton pour recommander
            if st.button(f"Voir livres similaires à : {row['title']}", key=f"rec_{idx}"):

                recs = recommend_books(row["title"])
                st.subheader(f"Livres similaires à : {row['title']}")

                if recs is None:
                    st.error("Pas de recommandations disponibles.")
                else:
                    st.table(recs)

