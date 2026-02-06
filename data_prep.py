import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin

# Load your labeled dataset
# Read without assuming headers first, then clean
df = pd.read_csv('data/labeled_companies.csv')
print("Raw loaded data:")
print(df.head())

# Check and drop duplicate header rows (e.g., if first row looks like headers)
# Assuming the duplicate is something like "company,website,label," in the first data row
if not df.empty and df.iloc[0]['company_name'].strip().lower() == 'company':
    df = df.drop(0).reset_index(drop=True)
    print("Dropped duplicate header row.")

# Ensure columns are named correctly (in case the CSV has no headers)
df.columns = ['company_name', 'website_url', 'label_relevance'] if len(df.columns) == 3 else df.columns
print("Cleaned data:")
print(df.head())

def scrape_website_with_links(url, max_pages=3):
    """
    Scrape main page + a few internal links (services/products/etc.)
    max_pages limits total pages scraped to avoid overloading.
    """
    scraped_texts = []
    visited = set()
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        queue = [url]

        while queue and len(visited) < max_pages:
            current_url = queue.pop(0)
            if current_url in visited:
                continue
            visited.add(current_url)

            response = requests.get(current_url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove scripts, styles, nav, header, footer
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()

            text = soup.get_text(separator=' ', strip=True)
            text = ' '.join(text.split())[:3000]
            if len(text) > 100:
                scraped_texts.append(text)

            # Add internal links containing keywords
            for a in soup.find_all('a', href=True):
                href = a['href']
                # Only follow internal links (same domain)
                full_url = urljoin(url, href)
                if url in full_url and any(k in href.lower() for k in ['service', 'product', 'solution']):
                    queue.append(full_url)

        return ' '.join(scraped_texts) if scraped_texts else None
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

# Scrape and add text column
df['scraped_text'] = None  # Initialize
for index, row in df.iterrows():
    url = row['website_url']
    if pd.notna(url):
        df.at[index, 'scraped_text'] = scrape_website_with_links(url)
        time.sleep(2)  # Delay to avoid rate limits (adjust as needed)

# Filter out rows with failed scrapes (scraped_text is None)
original_count = len(df)
df = df[df['scraped_text'].notna()]
removed_count = original_count - len(df)
print(f"Removed {removed_count} companies with failed scrapes.")

# Save with clean headers (no extra rows)
df.to_csv('labeled_companies_with_text.csv', columns=['company_name', 'website_url', 'label_relevance', 'scraped_text'], index=False)
print("Scraping complete. Check 'labeled_companies_with_text.csv'.")