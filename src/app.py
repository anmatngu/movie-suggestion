import streamlit as st
import joblib
import difflib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

movies_df = pd.read_csv('data\\movies.csv')

feature_extraction = joblib.load('src\\feature_extraction.pkl')
list_of_all_titles = movies_df['title'].tolist()
similarity = cosine_similarity(feature_extraction)

def movie_suggestion(user_input):
        input_movie = str(user_input)
        find_close_match = difflib.get_close_matches(input_movie, list_of_all_titles)
        print(f'Close match results: {find_close_match}')
        close_match = find_close_match[0]
        print(f'Closest match: {close_match}')
        index_of_the_movie = movies_df[movies_df['title'] == close_match]['index'].values[0]
        print(f'Index of the movie: {index_of_the_movie}')
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)
        print('Movies suggested for you : \n')

        i = 1
        max_suggestion = 10
        movie_list = 'You should watch '
        for movie in sorted_similar_movies:
            index = movie[0]
            title_from_index = movies_df[movies_df['index']==index]['title'].values[0]
            release_date_index = movies_df[movies_df['index']==index]['release_date'].values[0]
            if (i <= max_suggestion):
                movie_list += '\n' + str(i) + '. ' + title_from_index + ' - ' + f'({release_date_index})'
                i+=1
        return movie_list
def main():
    st.title("Movie Suggestion")
    movie_text = st.text_area("Enter movie name:", height=50)

    if st.button("Search"):
        suggestion = movie_suggestion(movie_text)
        st.write(suggestion)

if __name__ == '__main__':
    main()