from recommendation_system import RecommendationSystem
import pandas as pd

def main():
    df = pd.read_csv('tmdb_5000_movies.csv')
    df['profit'] = df['revenue'] - df['budget']
    df.drop(['homepage', 'id', 'status', 'tagline', 'vote_count', 'revenue'], axis = 1, inplace = True)

    movie_recommender = RecommendationSystem(df)
    movie_list = movie_recommender.get_recommendations('Man of Steel', 5)
    print(movie_list)


if __name__=='__main__':
    main()
