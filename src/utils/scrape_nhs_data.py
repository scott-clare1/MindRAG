import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd


class ScrapeNHS:
    def __init__(self, root_url: str, search_url: str):
        self.root_url = root_url
        self.search_url = search_url
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        }
        self.hrefs = []
        self.titles = []
        self.docs = []

    @staticmethod
    def _get_soup(url: str, headers: dict):
        page = requests.get(url, headers=headers)
        soup = BeautifulSoup(page.content, "html.parser")
        return soup

    @staticmethod
    def _build_document(p_tags):
        return "".join([p.text.replace("\xa0", " ") for p in p_tags])

    def _collect_data(self, url: str, title: str):
        condition_page = self._get_soup(url, self.headers)
        information = condition_page.find_all("section")
        for section in information:
            self.docs.append(self._build_document(section.find_all("p")))
            self.titles.append(title)
            self.hrefs.append(url)

    def __call__(self):
        url = self.root_url + self.search_url
        soup = self._get_soup(url, self.headers)
        div_content = soup.find(
            "div", class_="nhsuk-u-reading-width beta-hub-bottom-content"
        )
        conditions = div_content.find_all("a", href=True)
        for condition in tqdm(conditions):
            href = self.root_url + condition["href"]
            sub_soup = self._get_soup(href, self.headers)
            condition_contents = sub_soup.find(
                "ul", class_="nhsuk-hub-key-links beta-hub-key-links"
            )
            if condition_contents:
                condition_sublinks = condition_contents.find_all("a", href=True)
                for a_tag in condition_sublinks:
                    self._collect_data(a_tag["href"], a_tag.text)
            else:
                self._collect_data(href, condition.text)


if __name__ == "__main__":
    root = "https://www.nhs.uk"
    search_url = "/mental-health/conditions/"
    scraper = ScrapeNHS(root_url=root, search_url=search_url)
    scraper()
    data = pd.DataFrame(
        data={
            "title": scraper.titles,
            "documents": scraper.docs,
            "links": scraper.hrefs,
        }
    )
    data.to_csv("data/nhs_mental_health_data.csv")
