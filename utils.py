import os
import requests
import logging
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from concurrent.futures import ThreadPoolExecutor
import time
import google.generativeai as genai
import math
import numpy as np
from newspaper import Article
from transformers import pipeline
import faiss
from duckduckgo_search import DDGS
import shutil
import tempfile
import pandas as pd
import zipfile
import warnings
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom normalization for known edge cases
def normalize_country_name(name):
    SPECIAL_CASES = {
        "gambia, the": "gambia",
        "iran, islamic rep.": "iran",
        "yemen, rep.": "yemen"
    }
    name = name.lower().strip()
    return SPECIAL_CASES.get(name, name)

# -------------------- Load Country Mapping --------------------
def get_country_code_mapping():
    try:
        url = "https://api.worldbank.org/v2/sources/6/country?per_page=300&format=JSON"
        response = requests.get(url).json()
        rows = response["source"][0]["concept"][0]["variable"]
        mapping = {normalize_country_name(item["value"].lower()): item["id"] for item in rows}
        return mapping
    except Exception as e:
        logger.error(f"Error loading country code mapping: {e}")
        return {}

# -------------------- Extract Articles with Newspaper3k --------------------
def extract_article_content(url: str) -> str:
    """Fetch and extract the content of an article."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        logger.error(f"Error extracting article from {url}: {e}")
        return f"Error extracting article: {e}"

# -------------------- Fetch Top News Articles --------------------
def fetch_articles(country: str, query: str = "") -> list:
    """Uses DuckDuckGo to search for news articles related to a country and extracts the top 10 articles."""
    search_query = f"{country} economy debt {query}".strip()
    articles = []

    try:
        with DDGS() as ddgs:
            results = ddgs.text(search_query, max_results=15)
            for result in results:
                url = result.get("href") or result.get("url")
                if url:
                    content = extract_article_content(url)
                    if content and not content.startswith("Error"):
                        articles.append(content)
                if len(articles) >= 5:
                    break
    except Exception as e:
        logger.error(f"Error fetching articles for {country}: {e}")

    return articles

# -----------------Create Temporary Vectorstore for the news articles-----------------------------------
def create_temp_vectorstore_from_news(news_articles: list, embedding_model) -> FAISS:
    """
    Takes a list of news article texts, converts them into LangChain Documents,
    and builds a temporary FAISS vectorstore.
    """
    try:
        documents = [Document(page_content=article) for article in news_articles if article.strip()]
        if not documents:
            raise ValueError("No valid article content found.")
        
        temp_vectorstore = FAISS.from_documents(documents, embedding_model)
        return temp_vectorstore
    except Exception as e:
        logger.error(f"Error creating temporary vectorstore from news articles: {e}")
        return None

# -------------------- Extract Mentioned Country codes from Query --------------------
def extract_countries_from_query(query: str, mapping: dict) -> List[str]:
    mentioned = [mapping[name] for name in mapping if name in query.lower()]
    return mentioned

# -------------------- Extract Mentioned Country names from Query --------------------
def extract_countries_from_query_news(query: str, mapping: dict) -> List[str]:
    mentioned = [name for name in mapping if name in query.lower()]
    return mentioned

# -------------------- Load World Bank Vectorstore --------------------
def load_vectorstore_for_country_code(code: str, embedding_model) -> FAISS:
    path = os.path.join("/app/vectorstores", code)
    logger.info(f"📦 Looking for vectorstore at: {path}")
    
    if os.path.exists(path):
        logger.info(f"✅ Found vectorstore for {code}")
        try:
            return FAISS.load_local(path, embeddings=embedding_model, allow_dangerous_deserialization=True)
        except Exception as e:
            logger.error(f"Error loading vectorstore for {code}: {e}")
            return None
    else:
        logger.warning(f"❌ Vectorstore NOT found for {code}")
        return None

# --------- Load the IMF Vector Store --------------
def load_imf_vectorstore() -> RetrievalQA:
    try:
        excel_vectorstore = FAISS.load_local("/app/imf_excel_vectorstore/content/excel_vectorstore", 
                                            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
                                            allow_dangerous_deserialization=True)
        retriever = excel_vectorstore.as_retriever()
        llm = ChatGroq(
            temperature=0,
            model_name="gemma2-9b-it",  # or llama3-70b
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        qa_chain_imf = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
        return qa_chain_imf
    except Exception as e:
        logger.error(f"Error loading IMF vectorstore: {e}")
        return None
