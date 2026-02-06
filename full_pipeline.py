import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim
import time
import re
import random
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Load trained relevance model and vectorizer
model = joblib.load('relevance_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def get_company_website(company, driver):
    query = f"{company} official website"
    driver.get(f"https://www.google.com/search?q={query.replace(' ', '+')}")
    time.sleep(random.uniform(2.5, 5))  # human-like delay

    results = driver.find_elements(By.CSS_SELECTOR, "div.tF2Cxc")  # Google search results container
    bad_domains = [
        "facebook.com", "linkedin.com", "zoominfo.com",
        "yelp.com", "crunchbase.com", "opencorporates.com",
        "bloomberg.com"
    ]

    for r in results:
        try:
            link = r.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
            if any(b in link for b in bad_domains):
                continue
            return link  # first clean result
        except:
            continue

    return None




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

CA_CITIES = [
    "los angeles", "san diego", "san jose", "san francisco",
    "oakland", "irvine", "anaheim", "pasadena", "fremont",
    "santa clara", "santa ana", "riverside", "burbank"
]
# Function to check HQ via website
def check_hq_from_site(text):
    if not text: 
        return None
    
    text = text.lower()

    if re.search(r"\bca\b|\bcalifornia\b", text):
        return True

    for city in CA_CITIES:
        if city in text:
            return True

    if re.search(r"\b9\d{4}\b", text):  # CA ZIP heuristic
        return True

    return False



new_df = pd.read_csv('new_companies.csv')  # Assume this has the columns

def get_driver():
    options = Options()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)
    

# def scrape_revenue_google(company):
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

def normalize_company(name):
    name = name.replace("+", " ")
    name = name.replace("&", " and ")
    name = re.sub(r"\s+", " ", name)
    return name.strip()

def detect_acquisition(text):
    keywords = ['acquired by', 'merged with', 'taken over by', 'sold to', 'now part of']
    text_lower = text.lower()
    return 1 if any(keyword in text_lower for keyword in keywords) else 0




# Process each
results = []
options = Options()
driver = get_driver()

for index, row in new_df.iterrows():
    company = normalize_company(row['company_name'])
    url = get_company_website(company, driver)
    new_df.at[index, 'company_website'] = url
    print(url)
    # Scrape text (reuse your scrape function)
    scraped_text = scrape_website_with_links(url)
    if scraped_text:
        processed = preprocess_text(scraped_text)
        X_new = vectorizer.transform([processed])
        acquisition_detected = detect_acquisition(scraped_text)
        if acquisition_detected == 0:
            relevance_score = (model.predict_proba(X_new)[0][1]).round(2)  # Probability of relevant (1)
        else:
            relevance_score = 0
        
        print(f"Got text for {company}")
    else:
        relevance_score = 0.5  # No text = irrelevant
    
    # HQ and revenue checks
    # hq_ca = check_hq_from_site(scraped_text)

    options = Options()
    # options.add_argument("--headless=new")  # optional, if you want it to run in background, makes Chrome invisible
    # options.add_argument("--disable-blink-features=AutomationControlled")
    # driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    time.sleep(random.uniform(2.5, 5))  # avoid hitting too fast
    revenue = check_revenue_zoominfo(company,driver)
    print(company, "→", revenue)
    
    # Combined score (simple: relevance + checks)
    final_score = relevance_score * (1 if revenue else 0)
    
    results.append({
        'final_score': final_score,
        'company_name': company,
        'relevance_score': relevance_score,
        # 'hq_ca': hq_ca,
        'revenue': (revenue if revenue else 0),
        'website': url
        
    })

# Rank and save
output_df = pd.DataFrame(results).sort_values('final_score', ascending=False)
output_df.to_csv('ranked_companies.csv', index=False)
print("Pipeline complete. Check 'ranked_companies.csv' for rankings.")