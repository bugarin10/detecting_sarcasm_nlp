import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
import time
import difflib as dl


def getting_soup(movie_title, time_waiting=5):
    """
    This function takes in a movie title and returns the soup of the rotten tomatoes page for that movie.
    """
    try:
        # Start a headless browser (Chrome)
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")

        # Set up the WebDriver (make sure to replace 'path/to/chromedriver' with the actual path to your ChromeDriver executable)
        driver = webdriver.Chrome(options=options)

        # Navigate to the website
        driver.get(
            "https://www.rottentomatoes.com/"
        )  # Replace 'https://example.com' with the actual URL of the website

        # Locate the search input field by class and data-qa attribute

        search_input = driver.find_element(
            By.CSS_SELECTOR, 'input.search-text[data-qa="search-input"]'
        )

        # Type the movie name into the search input field
        search_input.send_keys(movie_title)

        # Submit the form (if needed, as some websites may require pressing Enter to submit the search)
        search_input.send_keys(Keys.RETURN)

        current_url = driver.current_url
        # Optionally, you can add a delay to see the result on the webpage (adjust the time.sleep value as needed)
        driver.get(current_url)

        # Wait for dynamic content to load (you may need to adjust the waiting time)
        time.sleep(time_waiting)

        # Get the page source after JavaScript execution
        page_source = driver.page_source

        # Parse the page source with BeautifulSoup
        soup = BeautifulSoup(page_source, "html.parser")
        # Close the browser
        driver.quit()

    except:
        soup = None
        print("Error with getting soup")
    return soup


def getting_movies_from_soup(soup):
    if soup is None:
        print("Soup was empty")
        return None
    """
    This function takes in a soup and returns a list of movies from the soup that match the name we are looking for.
    """
    try:
        media_rows = soup.select('search-page-media-row[data-qa="data-row"]')

        # Initialize a list to store the extracted information
        movie_list = {}

        # Loop through each search-page-media-row element and extract information
        for row in media_rows:
            cast = row["cast"]
            release_year = row["releaseyear"]
            tomatometer_score = row["tomatometerscore"]

            # Find the first <a> element with class 'unset' within the current search-page-media-row
            link_element = row.find("a", class_="unset")
            href = link_element["href"] if link_element else None
            key = href.split("/")[4]
            # Add the extracted information to the movie_list
            movie_list[key] = {
                "cast": cast,
                "release_year": release_year,
                "tomatometer_score": tomatometer_score,
                "href": href,
            }

    except:
        movie_list = None
        print("Error with getting movies from soup")

    return movie_list


def getting_formatted_name(movie: str, similiar_movies: dict) -> str:
    """
    This function takes in a movie title, and a dictionary of possible matches and returns the formatted name
    """

    try:
        movie_list = list(similiar_movies.keys())
        if len(movie_list) == 0:
            print("No movie found")
        else:
            return movie_list[0]
    except ValueError as e:
        print(e)
        return None


def getting_critics(movie_rt_format):
    """
    This function takes a movie in a rotentomatoes (rt)
    format and returns a list of the 20 first top-critics for that movie.
    """
    if movie_rt_format is not None:
        # Webscrapping the MLS Players Association Salary Guide
        url = (
            "https://www.rottentomatoes.com/m/"
            + movie_rt_format
            + "/reviews?type=top_critics"
        )

        # Send an HTTP GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, "html.parser")

            # Find the <div> with class 'review_table' and data-paginationmanager attribute
            review_div = soup.find(
                "div",
                class_="review_table",
                attrs={"data-paginationmanager": "paginatedDataContainer"},
            )

            # Check if the review_div is found before proceeding
            if review_div:
                # Find all <p> elements with class 'review-text' and data-qa attribute
                review_paragraphs = review_div.find_all(
                    "p", class_="review-text", attrs={"data-qa": "review-quote"}
                )

                # Extract and print the text content of each <p> element
                review_texts = [paragraph.text for paragraph in review_paragraphs]

                # Print the list of review texts
                # print(review_texts)
        else:
            review_texts = None
            print("Review div not found.")

        return review_texts
    else:
        print("The input in getting_critics was None")
        return None
