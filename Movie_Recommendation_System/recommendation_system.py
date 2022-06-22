import numpy as np
import pandas as pd
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


class RecommendationSystem:
    def __init__(self, df):
        self.df = df
        self.movie_docs = self.dataframe_to_string(df)
        self.X_train = []
        self.train_data(self.movie_docs)


    def json_to_string(self, df_item):
        temp = json.loads(df_item)
        str = ""
        for i in range(len(temp)):
            str = str + temp[i]["name"]
            if(i != len(temp) - 1):
                str = str + ", "
        return str


    def dataframe_to_string(self, df):
        movie_docs = []

        for i in range(df.shape[0]):
            movie_doc = f"Name of the movie is {df.iloc[i]['title']}. The movie was made with a badget of {df.iloc[i]['budget']} and earned \
            a profit of {df.iloc[i]['profit']}. The film comes under genre {self.json_to_string(df.iloc[i]['genres'])}. Keywprds that describe \
            the movie are {self.json_to_string(df.iloc[i]['keywords'])}. The movie was originally released in language {df.iloc[i]['original_language']}. \
            The film was released in languages {self.json_to_string(df.iloc[i]['spoken_languages'])}. Popularity score of the movie \
            is {df.iloc[i]['popularity']} with vote average of {df.iloc[i]['vote_average']}. The movie was made by \
            {self.json_to_string(df.iloc[i]['production_companies'])} production companies. It was filmed in {self.json_to_string(df.iloc[i]['production_countries'])}. \
            It was released on {df.iloc[i]['release_date']} with a runtime of {df.iloc[i]['runtime']}. {df.iloc[i]['overview']}"

            movie_docs.append(movie_doc)

        self.movie_docs = movie_docs
        return movie_docs


    def train_data(self, movie_docs):
        tfidf = TfidfVectorizer(norm="l2", stop_words='english', max_features=3500,
                use_idf=True, smooth_idf=True)
        X_train = tfidf.fit_transform(movie_docs)
        self.X_train = X_train
        return X_train


    def movie_to_index(self):
        movie_to_index = pd.Series(self.df.index, index = self.df['title'])
        return movie_to_index


    def get_recommendations(self, movie_name, n_recommendations):
        movie_index = self.movie_to_index()[movie_name]
        query = self.X_train[movie_index]
        query.toarray()

        scores = cosine_similarity(query, self.X_train)
        scores = scores.flatten()
        recommended_movie_index = (-scores).argsort()[0:n_recommendations]

        movie_list = self.df['title'].iloc[recommended_movie_index]
        movie_list = np.array(movie_list)
        return movie_list


    def get_recommendations_euclidean(self, movie_name, n_recommendations):
        movie_index = self.movie_to_index()[movie_name]
        query = self.X_train[movie_index]
        query.toarray()

        scores = euclidean_distances(query, self.X_train)
        scores = scores.flatten()
        recommended_movie_index = (-scores).argsort()[0:n_recommendations]

        movie_list = self.df['title'].iloc[recommended_movie_index]
        movie_list = np.array(movie_list)
        return movie_list