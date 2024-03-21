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

        if find_close_match:
            close_match = find_close_match[0]
            print(f'Closest match: {close_match}')
            index_of_the_movie = movies_df[movies_df['title'] == close_match]['index'].values[0]
            print(f'Index of the movie: {index_of_the_movie}')
            similarity_score = list(enumerate(similarity[index_of_the_movie]))
            sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)

            print('Movies suggested for you : \n')
            max_suggestion = 10
            movie_list = '## Movies Suggested for You:\n\n'

            for i, movie in enumerate(sorted_similar_movies[:max_suggestion], start=1):
                index = movie[0]
                title_from_index = movies_df[movies_df['index']==index]['title'].values[0]
                release_date_index = movies_df[movies_df['index']==index]['release_date'].values[0]
                similarity_score_value = movie[1]
                movie_list += f"{i}. **{title_from_index}** - ({release_date_index}) (Similarity Score: {similarity_score_value:.3f})\n\n"
            return movie_list
        else:
            return f"No close matches found for '{input_movie}'"
def main():
    st.markdown(
        """
        <style>
        .movie-suggestion {
            font-family: 'Arial', sans-serif;
            color: #333333;
            background-color: #f2f2f2;
            padding: 20px;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Movie Suggestion")

    user_input = st.text_input("Enter a movie name or keyword:")

    if user_input:
        suggestion = movie_suggestion(user_input)
        st.write(suggestion)

if __name__ == '__main__':
    main()