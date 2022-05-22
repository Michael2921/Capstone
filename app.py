import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


cartoon_data = pd.read_csv("cartoons.csv")
# Removing extraneous columns
cartoon_data.drop("Unnamed: 0", axis=1, inplace=True)


# Checking missing data
cartoon_data.isna().sum()

# 45 rows have missing data. We could fill up the rows or drop them. I chose to fill the rows with accurate data from the internet
# Shows still running are dropped

# Drop rows with shows that don't have end dates

cartoon_data = cartoon_data.drop(labels=[12,74,79,83,84, 98, 89, 122, 124, 140, 199, 219], axis=0)

# Create a list and fill the remaining na rows with its values

List=[2002,2003,1980,1980,1982,2021,1982,2011,2010,2020,2002,2007,2006,2005,2015,2003,2000,1967,1966,1969,2007,2012,2015,2021,2018,2020,2012,1993,2018,2020,2020,2017,2017]
cartoon_data.loc[cartoon_data.end_date.isna(),'end_date'] = List


# combining all the 5 selected features
combined_features = cartoon_data['cartoon_names']+' '+cartoon_data['release_date'].apply(str)+' '+cartoon_data['end_date'].apply(str)+' '+cartoon_data['genre']+' '+cartoon_data['imdb_score'].apply(str)

# Converting text data to feature vector
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

similarity = cosine_similarity(feature_vectors)

def recommend(movie):
    movie_index = cartoon_data[cartoon_data['cartoon_names'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommend_movies = []

    for i in movies_list:
        recommend_movies.append(cartoon_data.iloc[i[0]][0])  # Line of error
        print(recommend_movies)
    return recommend_movies


st.title('Movie Recommender System')

selected_movie_name = st.selectbox('Please select a show from the menu below', cartoon_data['cartoon_names'].values)


if st.button('Recommend'):
    recommendations = recommend(selected_movie_name)
    for i in recommendations:
        st.write(i)



