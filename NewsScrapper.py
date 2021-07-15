import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import pandas as pd


def get_data():
    df = pd.DataFrame(columns=['News', 'Target'])
    start = 1
    end = 501
    for i in range(start, end):  # iterating through web pages
        r = requests.get(f'https://www.politifact.com/factchecks/list/?page={i}', headers={"user-agent": user_agent.random})
        soup = BeautifulSoup(r.content, 'html.parser')
        headlines = soup.find_all('div', class_='m-statement__quote')

        images = soup.find_all('div', class_='m-statement__meter')

        for j in range(len(headlines)):  # iterating through different news headlines in a single webpage
            info = headlines[j].find('a').text
            info = info.replace("\n", "").replace('"', '').replace("'", "").replace(" ", " ")
            pic = images[j].find('div', class_='c-image')
            tar = pic.find('img', class_='c-image__thumb')  # extracting whether news is real or fake
            tar = tar.get('alt')
            if tar == "true":
                tar = 1
            else:
                tar = 0
            df = df.append({'News': info, 'Target': tar}, ignore_index=True)

        print(f'page {i} scraped')

        if i % 50 == 0:
            df.to_csv("fake_news_data1.csv", index=False)
            print(f"Checkpoint Save: {i} row/s saved")

    df.to_csv("fake_news_data1.csv", index=False)
    print(f"Final Save: {i} row/s saved")


if __name__ == "__main__":
    user_agent = UserAgent()
    get_data()
