import concurrent.futures
import json
from urllib.parse import urljoin, urlparse

from beam import Image, function


@function(image=Image().add_python_packages(["requests", "beautifulsoup4"]))
def scrape_page(url):
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(url)
    if response.status_code != 200:
        return {"url": url, "title": "", "content": "", "links": []}

    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.find(id="firstHeading").text
    content = soup.find(id="mw-content-text").find(class_="mw-parser-output")

    if not content:
        return {"url": url, "title": title, "content": "", "links": []}

    paragraphs = [p.text for p in content.find_all("p", recursive=False)]
    links = [urljoin(url, link["href"]) for link in content.find_all("a", href=True)]

    return {
        "url": url,
        "title": title,
        "content": "\n\n".join(paragraphs),
        "links": links,
    }


class WikipediaCrawler:
    def __init__(self, start_url, max_pages=100, batch_size=5):
        self.start_url = start_url
        self.max_pages = max_pages
        self.batch_size = batch_size
        self.visited_pages = set()
        self.pages_to_visit = [start_url]
        self.scraped_data = {}

    def is_wikipedia_url(self, url):
        parsed_url = urlparse(url)
        return parsed_url.netloc.endswith(
            "wikipedia.org"
        ) and parsed_url.path.startswith("/wiki/")

    def crawl(self):
        while len(self.scraped_data) < self.max_pages and self.pages_to_visit:
            # Create a batch of 5 pages to scrape that we have not yet visited
            batch = []
            while len(batch) < self.batch_size and self.pages_to_visit:
                p = self.pages_to_visit.pop(0)
                if p not in self.visited_pages:
                    batch.append(p)

            for result in scrape_page.map(batch):
                # Save the result and collect new links
                self.scraped_data[result["url"]] = result
                if len(self.scraped_data) < self.max_pages:
                    new_links = [
                        link
                        for link in result["links"]
                        if self.is_wikipedia_url(link)
                        and link not in self.visited_pages
                        and link not in self.pages_to_visit
                    ]
                    self.pages_to_visit.extend(new_links)

        print(f"Crawling completed. Scraped {len(self.scraped_data)} pages.")

    def get_scraped_data(self):
        return self.scraped_data


if __name__ == "__main__":
    start_url = "https://en.wikipedia.org/wiki/Web_scraping"
    crawler = WikipediaCrawler(start_url, max_pages=20)
    crawler.crawl()

    # Write the scraped data to a file
    with open("scraped_data.json", "w") as f:
        json.dump(crawler.get_scraped_data(), f)
