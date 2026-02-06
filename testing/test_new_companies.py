import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
import time

# Load trained model and vectorizer
model = joblib.load('relevance_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocess function (same as training)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Scrape function (reuse from your data_prep)
def scrape_website_with_links(url, max_pages=3):
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
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            text = soup.get_text(separator=' ', strip=True)
            text = ' '.join(text.split())[:3000]
            if len(text) > 100:
                scraped_texts.append(text)
            for a in soup.find_all('a', href=True):
                href = a['href']
                full_url = urljoin(url, href)
                if url in full_url and any(k in href.lower() for k in ['service', 'product', 'solution']):
                    queue.append(full_url)
        return ' '.join(scraped_texts) if scraped_texts else None
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

# Load new companies (CSV with company_name, website_url)
new_df = pd.read_csv('testing/test_new_companies.csv')  # Create this file

results = []
for index, row in new_df.iterrows():
    company = row['company_name']
    url = row['website_url']
    scraped_text = scrape_website_with_links(url)
    if scraped_text:
        processed = preprocess_text(scraped_text)
        X_new = vectorizer.transform([processed])
        relevance_prob = model.predict_proba(X_new)[0][1]  # Prob of relevant
        relevance_pred = model.predict(X_new)[0]
    else:
        relevance_prob = 0
        relevance_pred = 0
    
    results.append({
        'company_name': company,
        'predicted_relevance': relevance_pred,
        'relevance_probability': relevance_prob
    })

# Save results
output_df = pd.DataFrame(results)
output_df.to_csv('testing/test_predictions.csv', index=False)
print("Testing complete. Check 'testing/test_predictions.csv'.")