import pandas as pd
import numpy as np
from lib import *

movies = pd.read_csv("../00_data/oscarsWinners_10_22.csv")

critics = {}
counter = 0

for row in movies.iterrows():
    try:
        # Getting the soup after searching for a movie
        soup = getting_soup(row[1]["film"], time_waiting=5)
        # Getting all the movie titles
        similiar_movies = getting_movies_from_soup(soup)
        # Validation of the movie title
        movie_rt_format = list(similiar_movies.keys())[0]

        # Getting the critics
        critics[row[1]["film"]] = {
            "formatted_name": movie_rt_format,
            "top_critics": getting_critics(movie_rt_format),
        }
        counter += 1
        if counter % 10 == 0:
            print(f"{counter} movies processed")
    except:
        print(f"{row[1]['film']} Not Found")

# saving the critics
critics_df = pd.DataFrame(
    {
        "movie": list(critics.keys()),
        "formatted_name": list(map(lambda x: x["formatted_name"], critics.values())),
        "top_critics": list(map(lambda x: x["top_critics"], critics.values())),
    }
)

# merging the critics with the movies
movies = movies.merge(
    critics_df, left_on="film", right_on="movie", how="outer", indicator=True
)

# saving the movies
movies.to_csv("../00_data/oscarsWinners_10_22.csv", index=False)
