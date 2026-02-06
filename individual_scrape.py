import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim
import time
import re
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Load trained relevance model and vectorizer
model = joblib.load('relevance_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Function to preprocess text (same as training)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Scrape function (added from your data_prep code)
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

# Function to check HQ via Crunchbase (basic scraping; use API for production)
def check_hq_crunchbase(company_name):
    try:
        url = f"https://crunchbase.com/organization/{company_name.lower().replace(' ', '-')}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')  # Fixed: Added this line
        # Extract HQ (look for location text; customize based on site structure)
        hq_text = soup.find('span', class_='location-name')  # Example; inspect Crunchbase HTML
        if hq_text:
            geolocator = Nominatim(user_agent="company_tool")
            location = geolocator.geocode(hq_text.text)
            return 'CA' in location.raw.get('address', {}).get('state', '') if location else False
        return False
    except:
        return False

def check_revenue(company):
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    driver.get(f"https://www.bing.com/search?q={company}+revenue")
    time.sleep(3)  # let JS render

    text = driver.page_source.lower()
    driver.quit()

    m = re.search(
        rf"(what is|{company.lower()}'s)\s+revenue[^$]{{0,40}}"
        r"\$?\s*([0-9]+(?:\.[0-9]+)?)\s*(million|m|billion|b)",
        text
    )

    if not m:
        return None

    value = float(m.group(2))
    unit = m.group(3)

    if unit in ("b", "billion"):
        value *= 1000

    return value
# Load new companies (user input CSV: company_name, website_url)
new_df = pd.read_csv('new_companies.csv')  # Assume this has the columns










def get_driver():
    options = Options()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)

def scrape_revenue_google(company):
    driver = get_driver()
    try:
        driver.get("https://www.google.com")
        time.sleep(2)

        search_box = driver.find_element(By.NAME, "q")
        search_box.send_keys(f"{company} revenue")
        search_box.send_keys(Keys.RETURN)
        time.sleep(3)

        page_text = driver.find_element(By.TAG_NAME, "body").text

        revenue = extract_revenue(page_text)
        return revenue

    except Exception as e:
        print(f"Error scraping {company}: {e}")
        return None

    finally:
        driver.quit()

def check_revenue_zoominfo(company, driver):
    query = f"{company} revenue site:zoominfo.com"
    driver.get(f"https://www.google.com/search?q={query.replace(' ', '+')}")
    time.sleep(3)  # wait for results to load

    # Use current Google result container
    results = driver.find_elements(By.CSS_SELECTOR, "div.tF2Cxc")
    if not results:
        print("No results found")
        return None

    # Take the first result only
    first_result = results[0]
    try:
        link = first_result.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
        snippet = first_result.text.lower()  # snippet text

        # Extract ZoomInfo revenue
        m = re.search(
            r"revenue\s+is\s+\$?([0-9]+(?:\.[0-9]+)?)\s*(million|billion|m|b)",
            snippet, flags=re.IGNORECASE
        )

        if not m:
            return None

        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit in ("b", "billion"):
            val *= 1000  # convert billions to millions

        return val

    except Exception as e:
        print(f"Error extracting revenue: {e}")
        return None

def extract_revenue(text):
    patterns = [
        r"\$\s?\d+(?:\.\d+)?\s?(?:million|billion|m|bn)",
        r"USD\s?\d+(?:\.\d+)?\s?(?:million|billion|m|bn)",
        r"annual revenue[^$]{0,20}\$\s?\d+(?:\.\d+)?\s?(?:million|billion|m|bn)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)

    return None







# Process  
company = "VaCom Technologies"
url = "https://www.expresselectricalservices.com"
scraped_text = scrape_website_with_links(url)
if scraped_text:
    processed = preprocess_text(scraped_text)
    X_new = vectorizer.transform([processed])
    relevance_score = model.predict_proba(X_new)[0][1]  # Probability of relevant (1)
    print(f"Got text for {company}")
else:
    relevance_score = 0  # No text = irrelevant

# 1. Set up Chrome options
options = Options()
# options.add_argument("--headless=new")  # optional, if you want it to run in background, makes Chrome invisible
options.add_argument("--disable-blink-features=AutomationControlled")

# 2. Create the driver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
revenue_zoominfo = check_revenue_zoominfo(company, driver)
print(f"Zoominfo revenue: {revenue_zoominfo}")
driver.quit()