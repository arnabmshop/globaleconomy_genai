from langchain.tools import tool
from langchain.agents import create_react_agent
from langchain.prompts import PromptTemplate
from summarization_utils import (
    build_parallel_rag_model,
    summarize_with_openai,
    summarize_with_gemini,
    summarize_with_gemma,
    summarize_text,
    load_imf_vectorstore
)
from utils import (
    fetch_articles,
    get_country_code_mapping,
    extract_countries_from_query_news
)

# ---- TOOL 1: Hybrid RAG model ----
@tool("Hybrid Economic RAG Tool", return_direct=True)
def hybrid_rag_tool(query: str) -> str:
    """Answer complex economic questions using IMF data, World Bank country data, and economic news."""
    return build_parallel_rag_model(query)


# ---- TOOL 2: Gemini Summarizer ----
@tool("Gemini Summarizer", return_direct=True)
def gemini_summarizer_tool(input: str) -> str:
    """Summarize long economic content using Gemini."""
    return summarize_with_gemini(input, "Summarize this content")


# ---- TOOL 3: OpenAI Summarizer ----
@tool("OpenAI Summarizer", return_direct=True)
def openai_summarizer_tool(input: str) -> str:
    """Summarize economic content using OpenAI GPT-4."""
    return summarize_with_openai(input, "Summarize this content")


# ---- TOOL 4: Groq (Gemma) Summarizer ----
@tool("Gemma Summarizer via Groq", return_direct=True)
def gemma_summarizer_tool(input: str) -> str:
    """Summarize economic content using Gemma LLM via Groq."""
    return summarize_with_gemma(input, "Summarize this content")


# ---- TOOL 5: HuggingFace Summarizer ----
@tool("HuggingFace Text Summarizer", return_direct=True)
def huggingface_summarizer_tool(text: str) -> str:
    """Summarize text using Hugging Face summarization pipeline."""
    return summarize_text(text)


# ---- TOOL 6: News Insights Tool ----
@tool("Country News Insight", return_direct=True)
def country_news_tool(query: str) -> str:
    """Search and summarize economic news articles about a country."""
    try:
        mapping = get_country_code_mapping()
        country_names = extract_countries_from_query_news(query, mapping)
        if not country_names:
            return "No country found in query for news search."
        articles = fetch_articles(country_names[0], query)
        if not articles:
            return "No relevant news articles found."
        return summarize_text("\n\n".join(articles))
    except Exception as e:
        return f"❌ News Tool Error: {e}"


# ---- TOOL 7: IMF Data QA Tool ----
@tool("IMF Data Tool", return_direct=True)
def imf_data_tool(query: str) -> str:
    """Query IMF economic data using a vector search."""
    try:
        chain = load_imf_vectorstore()
        if not chain:
            return "IMF vectorstore could not be loaded."
        result = chain.invoke(query)
        return result["result"]
    except Exception as e:
        return f"❌ IMF Tool Error: {e}"


# 👇 Export tools list for agent use
tools = [
    hybrid_rag_tool,
    gemini_summarizer_tool,
    openai_summarizer_tool,
    gemma_summarizer_tool,
    huggingface_summarizer_tool,
    country_news_tool,
    imf_data_tool
]

# Define the prompt template for the agent
template = """You are an intelligent assistant that helps answer complex economic questions.
Your job is to decide which tool to use based on the user's query. Be sure to choose the most appropriate tool.

{query}
"""

# Create a prompt template
prompt = PromptTemplate(template=template, input_variables=["query"])

# Create the agent without specifying an LLM
agent = create_react_agent(
    tools=tools,
    prompt=prompt
)
