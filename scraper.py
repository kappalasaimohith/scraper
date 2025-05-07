import pandas as pd
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import re
import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import torch
import os

def load_models():
    model_name = "t5-small"
    local_cache_dir = os.path.join(os.getcwd(), "local_model_cache")

    os.makedirs(local_cache_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=local_cache_dir).to(device)

    device_index = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=device_index)
    return summarizer
summarizer = load_models()

def setup_chrome_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def get_flipkart_content(url):
    driver = setup_chrome_driver()
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.quit()
    return soup

def get_amazon_content(url):
    driver = setup_chrome_driver()
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.quit()
    return soup

def clean_price(value):
    clean_value = re.sub(r"[^\d.]", "", value)
    try:
        return float(clean_value)
    except:
        return None

def extract_amazon_data(soup):
    title = soup.select_one("#productTitle")
    price = soup.select_one(".a-price .a-offscreen")
    mrp = soup.select_one(".a-text-price .a-offscreen")
    rating = soup.select_one(".a-icon-alt")
    review_count = soup.select_one("#acrCustomerReviewText")
    description = soup.select_one("#productDescription")
    key_features = soup.select_one("#feature-bullets")

    price_val = clean_price(price.get_text()) if price else None
    mrp_val = clean_price(mrp.get_text()) if mrp else None

    discount = round(mrp_val - price_val, 2) if mrp_val and price_val else "N/A"
    discount_pct = round(((mrp_val - price_val) / mrp_val) * 100, 2) if mrp_val and price_val else "N/A"

    return {
        "Title": title.get_text(strip=True) if title else "N/A",
        "Price": f"₹{price_val}" if price_val else "N/A",
        "MRP": f"₹{mrp_val}" if mrp_val else "N/A",
        "Discount Amount": f"₹{discount}" if isinstance(discount, float) else "N/A",
        "Discount Percentage": f"{discount_pct}%" if isinstance(discount_pct, float) else "N/A",
        "Rating": rating.get_text(strip=True) if rating else "N/A",
        "Reviews": review_count.get_text(strip=True) if review_count else "N/A",
        "Description": description.get_text(strip=True) if description else "N/A",
        "Key Features": key_features.get_text(separator="\n", strip=True) if key_features else "N/A",
    }

def extract_flipkart_data(soup):
    title = soup.select_one("span.B_NuCI")
    price = soup.select_one("div._30jeq3._16Jk6d")
    mrp = soup.select_one("div._3I9_wc._2p6lqe")
    rating = soup.select_one("div._3LWZlK")
    review_count = soup.select_one("span._2_R_DZ")
    description = soup.select_one("div._1mXcCf")
    key_features = soup.select_one("div._2418kt ul")

    price_val = clean_price(price.get_text()) if price else None
    mrp_val = clean_price(mrp.get_text()) if mrp else None

    discount = round(mrp_val - price_val, 2) if mrp_val and price_val else "N/A"
    discount_pct = round(((mrp_val - price_val) / mrp_val) * 100, 2) if mrp_val and price_val else "N/A"

    return {
        "Title": title.get_text(strip=True) if title else "N/A",
        "Price": f"₹{price_val}" if price_val else "N/A",
        "MRP": f"₹{mrp_val}" if mrp_val else "N/A",
        "Discount Amount": f"₹{discount}" if isinstance(discount, float) else "N/A",
        "Discount Percentage": f"{discount_pct}%" if isinstance(discount_pct, float) else "N/A",
        "Rating": rating.get_text(strip=True) if rating else "N/A",
        "Reviews": review_count.get_text(strip=True) if review_count else "N/A",
        "Description": description.get_text(strip=True) if description else "N/A",
        "Key Features": key_features.get_text(separator="\n", strip=True) if key_features else "N/A",
    }

st.title("🔗 Ecommerce Product Scraper & Analyzer")

url = st.text_input("Enter a URL to summarize", "")

if url:
    try:
        st.info("Fetching content...")

        if "amazon" in url.lower():
            soup = get_amazon_content(url)
            extracted_info = extract_amazon_data(soup)
        elif "flipkart" in url.lower():
            soup = get_flipkart_content(url)
            extracted_info = extract_flipkart_data(soup)
        else:
            st.warning("The URL is neither an Amazon nor a Flipkart product page.")
            st.stop()

        st.write("Extracted Info:")
        st.write(extracted_info)

        main_content = "\n".join([f"{key}: {value}" for key, value in extracted_info.items()])
        clean_text = " ".join(main_content.split())
        truncated_text = clean_text[:512]  

        with st.expander("🔍 Show extracted text"):
            st.write(truncated_text)

        st.info("Summarizing...")
        summary = summarizer(truncated_text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']

        st.subheader("📝 Summary")
        st.write(summary)

        st.subheader("🛒 Product Details")
        product_details = {
            "Attribute": ["Title", "Price", "MRP", "Discount Percentage", "Discount Amount", "Description", "Rating", "Reviews"],
            "Value": [
                extracted_info.get("Title", "N/A"),
                extracted_info.get("Price", "N/A"),
                extracted_info.get("MRP", "N/A"),
                extracted_info.get("Discount Percentage", "N/A"),
                extracted_info.get("Discount Amount", "N/A"),
                extracted_info.get("Description", "N/A"),
                extracted_info.get("Rating", "N/A"),
                extracted_info.get("Reviews", "N/A"),
            ]
        }
        product_df = pd.DataFrame(product_details)
        st.dataframe(product_df, use_container_width=True)

        # Display key features
        if 'Key Features' in extracted_info:
            st.subheader("🔑 Key Features")
            for feature in extracted_info['Key Features'].split("\n"):
                st.write(f"- {feature}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
