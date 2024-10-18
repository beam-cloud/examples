import concurrent.futures
import json
from urllib.parse import urljoin, urlparse

from beam import Image, function


@function(image=Image().add_python_packages(["requests", "beautifulsoup4"]))
def scrape_page(url):
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.find(id="firstHeading").text
        content = soup.find(id="mw-content-text").find(class_="mw-parser-output")

        paragraphs = []
        new_links = []

        if content:
            paragraphs = [p.text for p in content.find_all("p", recursive=False)]

            # Extract links
            links = content.find_all("a", href=True)
            new_links = [urljoin(url, link["href"]) for link in links]

        return {
            "url": url,
            "title": title,
            "content": "\n\n".join(paragraphs),
            "links": new_links,
        }
    return None


class WikipediaCrawler:
    def __init__(self, start_url, max_pages=100):
        self.start_url = start_url
        self.max_pages = max_pages
        self.visited_pages = set()
        self.pages_to_visit = [start_url]
        self.scraped_data = {}

    def is_wikipedia_url(self, url):
        parsed_url = urlparse(url)
        return parsed_url.netloc.endswith(
            "wikipedia.org"
        ) and parsed_url.path.startswith("/wiki/")

    def crawl(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {}
            while len(self.scraped_data) < self.max_pages and (
                self.pages_to_visit or future_to_url
            ):
                while (
                    len(future_to_url) < 5
                    and self.pages_to_visit
                    and len(self.scraped_data) < self.max_pages
                ):
                    url = self.pages_to_visit.pop(0)
                    if url not in self.visited_pages:
                        self.visited_pages.add(url)
                        future = executor.submit(scrape_page.remote, url)
                        future_to_url[future] = url

                if future_to_url:
                    done, _ = concurrent.futures.wait(
                        future_to_url, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for future in done:
                        url = future_to_url.pop(future)
                        try:
                            result = future.result()
                            if result and len(self.scraped_data) < self.max_pages:
                                self.scraped_data[result["url"]] = result
                                if len(self.scraped_data) < self.max_pages:
                                    new_links = [
                                        link
                                        for link in result["links"]
                                        if self.is_wikipedia_url(link)
                                        and link not in self.visited_pages
                                        and link not in self.pages_to_visit
                                        and link
                                        not in [future_to_url[f] for f in future_to_url]
                                    ]
                                    self.pages_to_visit.extend(new_links)
                        except Exception as e:
                            print(f"Error processing {url}: {str(e)}")

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
