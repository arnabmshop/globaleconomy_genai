import os
import requests
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings
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
from langchain.docstore.document import Document
import shutil
import tempfile
import pandas as pd
import zipfile
import warnings
from langchain.chat_models import ChatOpenAI
from utils import load_vectorstore_for_country_code,get_country_code_mapping, extract_countries_from_query_news,fetch_articles, extract_countries_from_query, create_temp_vectorstore_from_news,load_imf_vectorstore
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Set your Gemini API key securely
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# Load Gemini model
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

# Summarizer pipeline
summarizer = pipeline("summarization")

#-----------Loading all the batches of countries--------
selected_codes_x = [folder for folder in os.listdir("/app/vectorstores/") if os.path.isdir(os.path.join("/app/vectorstores/", folder))]

# Split countries into smaller batches for parallel processing
max_countries_per_batch = 10  # Adjust based on your token limits
num_batches = math.ceil(len(selected_codes_x) / max_countries_per_batch)

print(f"📊 Number of batches to process: {num_batches}")

batches = [selected_codes_x[i * max_countries_per_batch: (i + 1) * max_countries_per_batch] for i in range(num_batches)]

# Debugging: Print the batches
print(f"Batch split into: {batches}")

#------------------- Preload all vectorstores for required country codes------------------------
from collections import defaultdict
batch_vectorstores_dict = defaultdict(list)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

for i, batch in enumerate(batches):
    for country_code in batch:
        vectorstore_path = os.path.join("/app/vectorstores/", country_code)
        if os.path.isdir(vectorstore_path):
            try:
                vs = load_vectorstore_for_country_code(country_code, embedding_model)
                if vs:
                    print(f"✅ Preloaded vectorstore for '{country_code}'")
                    batch_vectorstores_dict[i].append(vs)
                else:
                    print(f"❌ Failed to load vectorstore for '{country_code}'")
            except Exception as e:
                print(f"❌ Error loading vectorstore for '{country_code}': {e}")

# -------------------- Summarize Articles --------------------
def summarize_text(text: str) -> str:
    """Summarize the text using Hugging Face's summarization pipeline."""
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

#----------------Summarizing using Gemini model---------------------------------------------------------------------------
def summarize_with_gemini(context: str, query: str) -> str:
    try:
        prompt = f"""
You are an expert economic analyst. Based on the data provided below, answer the user's question as accurately as possible. The data may be partial, contradictory, or noisy and may come from multiple sources (e.g., IMF, World Bank, news articles, etc.).

Your task is to:
- Analyze all available information relevant to the user's query.
- If a data source is irrelevant to the question, ignore it completely — do not mention it or say that it lacks data.
- Provide the best possible or approximate answer along with the reason based on what's available even if the claims are conflicting.
- If a definitive answer cannot be given, explain what kind of data would help further — but do not blame specific sources.
- Ensure your answer is clear, respectful, concise, and well-reasoned.

User query:
{query}

Data:
{context}

Respond thoughtfully below:
"""
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Gemini summarization failed: {e}"
    
def summarize_with_gemma(context: str, query: str) -> str:
    # Initialize Gemma model via Groq
    llm = ChatGroq(
    temperature=0,
    model_name="gemma2-9b-it",  # or llama3-70b
    groq_api_key=os.getenv("GROQ_API_KEY")
)
    try:
        prompt = f"""
You are an expert economic analyst. Based on the data provided below, answer the user's question as accurately as possible. The data may be partial, contradictory, or noisy and may come from multiple sources (e.g., IMF, World Bank, news articles, etc.).

Your task is to:
- Analyze all available information relevant to the user's query.
- If a data source is irrelevant to the question, ignore it completely — do not mention it or say that it lacks data.
- Provide the best possible or approximate answer along with the reason based on what's available even if the claims are conflicting.
- If a definitive answer cannot be given, explain what kind of data would help further — but do not blame specific sources.
- Ensure your answer is clear, respectful, concise, and well-reasoned.

User query:
{query}

Data:
{context}

Respond thoughtfully below:
"""
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"❌ Gemma summarization failed: {e}"
    
def summarize_with_openai(context: str, query: str) -> str:
    # Initialize OpenAI
    llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-4",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
    try:
        prompt = f"""
You are an expert economic analyst. Based on the data provided below, answer the user's question as accurately as possible. The data may be partial, contradictory, or noisy and may come from multiple sources (e.g., IMF, World Bank, news articles, etc.).

Your task is to:
- Analyze all available information relevant to the user's query.
- If a data source is irrelevant to the question, ignore it completely — do not mention it or say that it lacks data.
- Provide the best possible or approximate answer along with the reason based on what's available even if the claims are conflicting.
- If a definitive answer cannot be given, explain what kind of data would help further — but do not blame specific sources.
- Ensure your answer is clear, respectful, concise, and well-reasoned.

User query:
{query}

Data:
{context}

Respond thoughtfully below:
"""
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"❌ OpenAI summarization failed: {e}"
    
# -------------------- Load IMF Vectorstore --------------------
def load_imf_vectorstore():
    try:
        excel_vectorstore = FAISS.load_local(
            "/app/imf_excel_vectorstore/content/excel_vectorstore",
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            allow_dangerous_deserialization=True
        )
        retriever = excel_vectorstore.as_retriever()
        llm = ChatGroq(
            temperature=0,
            model_name="gemma2-9b-it",  # or llama3-70b
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        qa_chain_imf = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    except Exception as e:
        print(f"❌ Error loading vectorstore: {e}")
        excel_vectorstore = None  # or fallback logic
    return qa_chain_imf

# -------------------- Process Country Batch --------------------
# def process_country_batch(batch_number,countries_batch, query, embedding_model, llm):

def process_country_batch(batch_number, query, vectorstores_batches, embedding_model, llm):

    # Merge all retrievers
    if not vectorstores_batches:
        return []

    if len(vectorstores_batches) == 1:
        retriever = vectorstores_batches[0].as_retriever()
    else:
        from langchain.retrievers import EnsembleRetriever
        retrievers = [vs.as_retriever() for vs in vectorstores_batches]
        retriever = EnsembleRetriever(retrievers=retrievers, weights=[1] * len(retrievers))

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    try:
        result = qa_chain.invoke(query)
        return result["result"]
    except Exception as e:
        #print(f"⚠️ Groq model failed for batch {batch_number}: {e}. Falling back to Gemini summarization.")
        try:
            all_docs = []
            for vs in vectorstores_batches:
                all_docs.extend(vs.similarity_search("", k=10))
            combined_context = "\n\n".join([doc.page_content for doc in all_docs])
            return summarize_with_openai(combined_context, query)
        except Exception as fallback_error:
            return f"❌ Unable to summarize batch due to fallback failure: {fallback_error}"
        
# -------------------- Build Hybrid RAG Model with Parallel Processing --------------------
def build_parallel_rag_model(query: str):
    # Init LLM and embeddings
    llm = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",  # or llama3-70b
    groq_api_key=os.getenv("GROQ_API_KEY")
)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Country name → code mapping
    name_to_code = get_country_code_mapping()

    # Define the name you used when saving the vectorstore
    vectorstore_name = "imf_excel_vectorstore"  # Replace with the actual name you used

    # Try to detect mentioned countries
    selected_codes = extract_countries_from_query(query, name_to_code)

    #Querying the IMF excel dataset
    # Load the IMF vectorstore and get the RetrievalQA chain
    qa_chain_imf = load_imf_vectorstore()
    response_imf = qa_chain_imf.invoke(query)
    # Check if any documents were actually retrieved
    if not response_imf.get("source_documents") or not response_imf["source_documents"]:
                answer_imf= "Source IMF: Nothing"
    else:
                answer_imf = "Source IMF: " + response_imf["result"]

    # If no countries detected, use all
    if not selected_codes:
        print("🔎 No countries explicitly found in query. Searching across all vectorstores...")

        selected_codes =  selected_codes_x

        # Process each batch in parallel
        with ThreadPoolExecutor() as executor:
            #futures = [executor.submit(process_country_batch, i, batch, query, embedding_model, llm) for i,batch in enumerate(batches)]
            futures = [executor.submit(process_country_batch, i, query, batch_vectorstores_dict[i], embedding_model, llm) for i in range(len(batches))]
            results = [future.result() for future in futures]

        # Debugging: Check if results are empty
        print(f"Results from batch processing: {results}")

        # Merge results from all batches
        merged_results = []
        for batch_result in results:
            if batch_result:
                if isinstance(batch_result, list):
                    merged_results.extend(batch_result)
                elif isinstance(batch_result, str):
                    merged_results.append(batch_result)

        # If merged results are empty, handle that case
        if not merged_results:
            return "❌ No results found after processing batches."

        # Convert merged results to a list of Documents
        documents = [Document(page_content=result) for result in merged_results]

        # Create a new vectorstore or retriever using these documents
        merged_vectorstore = FAISS.from_documents(documents, embedding_model)

        # Now process the query on the merged data with retry handling
        max_retries = 3
        retry_delay = 10  # seconds

        for attempt in range(max_retries):
            try:
                combined_context = "\n\n".join([r for r in merged_results if isinstance(r, str) and r.strip()])
                final_answer = summarize_with_gemini(combined_context, query)
                # Combine both answers
                combined_final_answer = final_answer + "\n\n" + answer_imf
                break  # Exit loop if successful
            except Exception as e:
                error_msg = str(e)
                if "rate limit" in error_msg.lower() or "Rate limit" in error_msg:
                    wait_time = retry_delay * (attempt + 1)
                    print(f"⚠️ Rate limit hit. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Error: {error_msg}")
                    final_answer = f"❌ Error: {e}"
                    break

        # Always delete the vectorstore to free memory
        try:
            del merged_vectorstore
        except:
            pass

        return combined_final_answer

    else:
        # If specific countries are mentioned, proceed normally
        vectorstores = [load_vectorstore_for_country_code(code, embedding_model) for code in selected_codes]
        vectorstores = [vs for vs in vectorstores if vs is not None]
        country_name = extract_countries_from_query_news(query, name_to_code)
        news_articles = fetch_articles(country_name)
        news_vectorstore = create_temp_vectorstore_from_news(news_articles, embedding_model)

        # Add the news vectorstore if it exists
        if news_vectorstore:
            vectorstores.append(news_vectorstore)

        # Merge all retrievers
        if not vectorstores:
            return "❌ No valid country vectorstores found."
        elif len(vectorstores) == 1:
            retriever = vectorstores[0].as_retriever()
        else:
             # Merge all vectorstores into one
            merged_vectorstore = vectorstores[0]
            for vs in vectorstores[1:]:
                 merged_vectorstore.merge_from(vs)
            retriever = merged_vectorstore.as_retriever()
            #from langchain.retrievers import EnsembleRetriever
            #retriever = EnsembleRetriever(retrievers=[vs.as_retriever() for vs in vectorstores], weights=[1]*len(vectorstores))

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        # Process the query and return the result
        try:
            result = qa_chain.invoke(query)
            final_answer = result["result"]
            # Combine both answers
            combined_final_answer = final_answer + "\n\n" + answer_imf
            return combined_final_answer

        except Exception as e:
            if hasattr(e, 'response') and e.response.status_code == 429:
                partial = getattr(e, 'partial_text', None)
                if partial:
                    return partial
            return f"❌ Error: {e}"
