import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cartoon_data = pd.read_csv("cartoons.csv")

cartoon_data.rename(columns = {'Unnamed: 0':'index'}, inplace = True)


# combining all the 5 selected features
combined_features = cartoon_data['cartoon_names'] + ' ' + cartoon_data['release_date'].apply(str) + ' ' + cartoon_data[
    'end_date'].apply(str) + ' ' + cartoon_data['genre'] + ' ' + cartoon_data['imdb_score'].apply(str)

# Converting text data to feature vector
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

similarity = cosine_similarity(feature_vectors)


def recommend(movie):
    movie_index = cartoon_data[cartoon_data['cartoon_names'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
  #  print(distances)

    recommend_movies = []

    for i in movies_list:
        recommend_movies.append(cartoon_data.iloc[i[0]][1])
      #  print(recommend_movies)

    x1 = movies_list[0][0]  # starts here
    y1 = movies_list[0][1]
    x2 = movies_list[1][0]
    y2 = movies_list[1][1]
    x3 = movies_list[2][0]
    y3 = movies_list[2][1]
    x4 = movies_list[3][0]
    y4 = movies_list[3][1]
    x5 = movies_list[4][0]
    y5 = movies_list[4][1]

    namex1 = cartoon_data['cartoon_names'].values[x1]
    namex2 = cartoon_data['cartoon_names'].values[x2]
    namex3 = cartoon_data['cartoon_names'].values[x3]
    namex4 = cartoon_data['cartoon_names'].values[x4]
    namex5 = cartoon_data['cartoon_names'].values[x5]



    plotdata = pd.DataFrame({"pies": [y1, y2, y3, y4, y5]},
                            index=[namex1, namex2, namex3, namex4, namex5])

    plotdata.plot(kind="barh")
    st.text("Visualizations describing movie similarity")
    st.text('Bar chart')
    st.bar_chart(plotdata, height=300)
    st.text('Line chart')
    st.line_chart(plotdata, height=300)
    st.text('Area chart')
    st.area_chart(plotdata, height=300)

    return recommend_movies


st.title('Movie Recommender System')

#cartoon_data.head(100)["release_date"].plot()

selected_movie_name = st.selectbox('Please select a show from the menu below', cartoon_data['cartoon_names'].values)

if st.button('Recommend'):

    recommendations = recommend(selected_movie_name)
    st.text('Here are the results from most similar to least similar')
    for i in recommendations:
        st.write(i)
