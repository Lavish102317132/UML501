import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

def scrape_books():
    print("Scraping books.toscrape.com ...")
    base_url = "https://books.toscrape.com/catalogue/page-{}.html"
    books = []
    for page in range(1, 51):
        url = base_url.format(page)
        response = requests.get(url)
        if response.status_code != 200:
            break
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('article', class_='product_pod')
        for book in articles:
            title = book.h3.a['title']
            price = book.find('p', class_='price_color').text.strip()
            availability = book.find('p', class_='instock availability').text.strip()
            rating = book.p['class'][1]
            books.append({
                "Title": title,
                "Price": price,
                "Availability": availability,
                "Star Rating": rating
            })
    df_books = pd.DataFrame(books)
    df_books.to_csv("books.csv", index=False)
    print("✅ books.csv saved successfully.\n")

def scrape_imdb():
    print("Scraping imdb.com/chart/top ...")
    url = "https://www.imdb.com/chart/top/"
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(3)
    movies = []
    rows = driver.find_elements("css selector", "tbody.lister-list tr")
    for row in rows:
        rank = row.find_element("css selector", ".titleColumn").text.split('.')[0]
        title = row.find_element("css selector", ".titleColumn a").text
        year = row.find_element("css selector", ".secondaryInfo").text.strip("()")
        rating = row.find_element("css selector", ".imdbRating strong").text
        movies.append({
            "Rank": rank,
            "Movie Title": title,
            "Year of Release": year,
            "IMDB Rating": rating
        })
    driver.quit()
    df_imdb = pd.DataFrame(movies)
    df_imdb.to_csv("imdb_top250.csv", index=False)
    print("✅ imdb_top250.csv saved successfully.\n")

def scrape_weather():
    print("Scraping timeanddate.com/weather ...")
    base_url = "https://www.timeanddate.com/weather/"
    countries = ["usa", "india", "uk", "australia", "canada"]
    cities = ["new-york", "delhi", "london", "sydney", "toronto"]
    weather_data = []
    for country, city in zip(countries, cities):
        url = f"{base_url}{country}/{city}"
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")
        city_name = soup.find("h1").text.strip()
        temp = soup.find("div", class_="h2").text.strip() if soup.find("div", class_="h2") else "N/A"
        condition = soup.find("div", class_="small").text.strip() if soup.find("div", class_="small") else "N/A"
        weather_data.append({
            "City Name": city_name,
            "Temperature": temp,
            "Weather Condition": condition
        })
    df_weather = pd.DataFrame(weather_data)
    df_weather.to_csv("weather.csv", index=False)
    print("✅ weather.csv saved successfully.\n")

if __name__ == "__main__":
    scrape_books()
    scrape_imdb()
    scrape_weather()
    print("🎯 All scraping tasks completed successfully!")
