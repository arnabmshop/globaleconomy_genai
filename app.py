import streamlit as st
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()

from utils import (
    load_vectorstore_for_country_code,get_country_code_mapping, extract_countries_from_query_news,
    fetch_articles, extract_countries_from_query, create_temp_vectorstore_from_news,
    load_imf_vectorstore
)
from summarization_utils import * 
from langchain.embeddings import HuggingFaceEmbeddings


# Import the tools and agent executor from tools.py
from tools import agent,tools  # Import the agent created in tools.py
from langchain.agents import AgentExecutor

# --------------------- Streamlit UI ---------------------
st.set_page_config(page_title="🌍 Global Economic Insight", layout="wide")

st.title("🌍 Global Economic Insight Generator")
st.markdown("Ask anything about a country's debt, economy, or financial indicators. Based on data open sourced from World Bank, IMF and Online News Articles.")

query = st.text_input("🧠 Enter your question here:")
run_button = st.button("Analyze")

# Initialize the AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools)

if run_button and query.strip():
    with st.spinner("🔎 Fetching data and generating insights..."):
        try:
            #response = build_parallel_rag_model(query)
            response = agent_executor.invoke({"input": query})
            st.success("✅ Analysis Complete")
            st.markdown("### 📊 Answer:")
            st.write(response)
        except Exception as e:
            st.error(f"❌ Error: {e}")
