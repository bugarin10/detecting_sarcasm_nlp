import pandas as pd
from bs4 import BeautifulSoup
import requests
import csv


def url_request(year):

    url_base = "https://en.wikipedia.org/wiki/List_of_American_films_of_"
    url = url_base + year
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    return soup

def movie_scrape(soup, year):
     # Locate Table
    content_div = soup.find('div', id='mw-content-text')
    output_div = content_div.find('div', class_='mw-content-ltr mw-parser-output')
    tables = output_div.find_all('table', class_='wikitable sortable')
    print(len(tables))
    title_list = []
    # Iterate through tables
    for table in tables:
        body = table.find('tbody')
        rows = body.find_all('tr')
        # Iterate through rows

        for row in rows:
            title_found = False
            cells = row.find_all('td')
            for cell in cells:
                link = cell.find('a')
                if link and link.get('title'):
                    title_list.append(link.get('title'))
                    title_found = True
                    break
            if not title_found:
                print(f'Title not found at row {len(title_list)+1}')
    cleaned_list = [s.replace("(film)", "") for s in title_list]
    cleaned_list = [s.replace(f"({year} film)", "") for s in cleaned_list]
    return cleaned_list

def save_csv(file_name, movies_dict):

    # Open the file in write mode
    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        # Create a csv writer object
        writer = csv.writer(file)
        
        # Write the header row
        writer.writerow(['Year', 'Movie Title'])
        
        # Iterate over the dictionary and write rows to the CSV file
        for year, movies in movies_dict.items():
            for movie in movies:
                writer.writerow([year, movie])

    print(f"Dictionary has been converted to CSV file named {file_name}.")

if __name__ == "__main__":
    movie_titles = {}
    year_list = [str(x) for x in range(2000,2023)]
    for year in year_list:
        print(year)
        soup = url_request(year)
        titles = movie_scrape(soup, year)
        movie_titles[f'{year}'] = titles
    file_name = 'movieTitles_00-22.csv'
    save_csv(file_name, movie_titles)


