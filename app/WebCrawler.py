import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from abc import ABC, abstractmethod
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

class LinkParsingStrategy(ABC):
    @abstractmethod
    def parse_links(self, base_url, html_content):
        pass

class WebCrawler:
    def __init__(self, base_url, link_parsing_strategy: LinkParsingStrategy):
        self.base_url = base_url
        self.visited_urls = set()
        self.link_parsing_strategy = link_parsing_strategy

    def fetch_data(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def crawl(self, url):
        if url in self.visited_urls:
            return
        print(f"Crawling: {url}")
        self.visited_urls.add(url)

        html_content = self.fetch_data(url)
        if html_content is None:
            return

        links = self.link_parsing_strategy.parse_links(self.base_url, html_content)
        for link in links:
            self.crawl(link)

        tables = self.table_parsing_strategy.parse_tables(html_content)
        print(json.dumps(tables, indent=2))  # Print the extracted tables as JSON

class InternalLinksStrategy(LinkParsingStrategy):
    def parse_links(self, base_url, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        links = set()
        for anchor in soup.find_all('a', href=True):
            link = urljoin(base_url, anchor['href'])
            if link.startswith(base_url):
                links.add(link)
        return links

class AllLinksStrategy(LinkParsingStrategy):
    def parse_links(self, base_url, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        links = set()
        for anchor in soup.find_all('a', href=True):
            link = urljoin(base_url, anchor['href'])
            links.add(link)
        return links

class ConfluenceStrategy(ABC):
    def parse_tables(self, html_content):
        """
        Extract Confluence tables from HTML content and return as JSON-like structure.

        Args:
            html_content (str): HTML content containing Confluence tables
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = []

        # Find all Confluence tables (typically have 'confluenceTable' class)
        for table in soup.find_all('table', class_='confluenceTable'):
            table_data = {
                'headers': [],
                'rows': []
            }

            # Extract headers (th elements)
            header_row = table.find('tr')
            if header_row:
                table_data['headers'] = [th.get_text(strip=True)
                                       for th in header_row.find_all(['th', 'td'])]

            # Extract table rows
            for row in table.find_all('tr')[1:]:  # Skip header row
                cells = row.find_all(['td', 'th'])
                row_data = {
                    'cells': [cell.get_text(strip=True) for cell in cells],
                    'links': [cell.a.get('href') for cell in cells if cell.a]
                }
                table_data['rows'].append(row_data)

            tables.append(table_data)

        return tables

    def extract_headings_and_links(self, html_content, pageId, directory="confluence_responses3"):
        soup = BeautifulSoup(html_content, 'html.parser')
        data = []
        child_page_ids = set()
        os.makedirs(directory, exist_ok=True)

        headings = soup.find_all(['h1', 'p'])
        for heading in headings:
            if heading.name == 'h1' or (heading.name == 'p' and heading.find('strong')):
                data.append({'type': 'heading', 'text': heading.get_text(strip=True)})

        links = soup.find_all('a')
        for link in links:
            link_data = {
                'type': 'link',
                'text': link.get_text(strip=True),
                'url': link.get('href'),
                'data-linked-resource-id': link.get('data-linked-resource-id'),
                'data-linked-resource-type': link.get('data-linked-resource-type'),
            }
            data.append(link_data)
            if link_data['data-linked-resource-id']:
                child_page_ids.add(link_data['data-linked-resource-id'])

        confluence_tables = self.parse_tables(html_content)
        data.append({'type': 'confluence_tables', 'text': confluence_tables})

        file_name = f"{pageId}_html.json"
        file_path = os.path.join(directory, file_name)

        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump({'data': data, 'page_id': pageId, 'child_page_ids': list(child_page_ids)}, json_file, indent=4)

        return data, child_page_ids

class WebCrawlerWorker:
    def __init__(self, base_url, link_parsing_strategy):
        self.base_url = base_url
        self.link_parsing_strategy = link_parsing_strategy

    def fetch_and_process(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            html_content = response.text
            links = self.link_parsing_strategy.parse_links(self.base_url, html_content)
            return links, html_content
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return set(), None

class WebCrawlerManager:
    def __init__(self, base_url, link_parsing_strategy, max_workers=5):
        self.base_url = base_url
        self.link_parsing_strategy = link_parsing_strategy
        self.visited_urls = set()
        self.to_visit_urls = set([base_url])
        self.max_workers = max_workers

    def crawl(self):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while self.to_visit_urls:
                futures = {executor.submit(self.crawl_url, url): url for url in self.to_visit_urls}
                self.to_visit_urls.clear()

                for future in as_completed(futures):
                    url = futures[future]
                    try:
                        links, html_content = future.result()
                        print(f"Links: {links}")
                        print(f"HTML Content: {html_content}")
                        self.visited_urls.add(url)
                        self.to_visit_urls.update(links - self.visited_urls)
                        # Process the HTML content if needed
                        print(f"Crawled: {url}")
                    except Exception as e:
                        print(f"Error processing {url}: {e}")

    def crawl_url(self, url):
        worker = WebCrawlerWorker(self.base_url, self.link_parsing_strategy)
        return worker.fetch_and_process(url)

