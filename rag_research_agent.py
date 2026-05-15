import os
import streamlit as st

# Load Streamlit Cloud secrets into environment variables
try:
    for k, v in st.secrets.items():
        os.environ[k] = v
except Exception:
    pass

from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
import requests
from html.parser import HTMLParser

# Additional imports for embeddings and FAISS
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

torch.set_default_device('cpu')

# -------------------- ENV + GEMINI --------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# -------------------- Helper: strip HTML --------------------
class Stripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.data=[]
    def handle_data(self,d): self.data.append(d)
    def get_data(self): return ' '.join(self.data)

def strip_tags(html):
    s=Stripper()
    s.feed(html)
    return s.get_data()

# -------------------- Search local PDF --------------------
def search_documents(query, pdf_file):
    if pdf_file is None: 
        return []
    reader=PdfReader(pdf_file)
    chunks=[]
    for i,page in enumerate(reader.pages):
        text=page.extract_text()
        if text:
            # Chunk text into smaller pieces of max 500 characters for embeddings
            for j in range(0, len(text), 500):
                chunk_text = text[j:j+500]
                chunks.append({"text": chunk_text, "source": pdf_file.name, "page": i+1})
    return chunks

# -------------------- Search the web (Google scraping only) --------------------
import logging

def search_web(query, top_k=3):
    import logging
    results = []
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        logging.error("SERPAPI_API_KEY not set in environment variables")
    try:
        from serpapi import GoogleSearch
        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": top_k
        }
        search = GoogleSearch(params)
        data = search.get_dict()
        organic_results = data.get("organic_results", [])
        for result in organic_results[:top_k]:
            snippet = result.get("snippet", "")
            link = result.get("link", "")
            if snippet and link:
                results.append({"text": snippet, "source": link, "page": None})
    except ImportError:
        logging.error("serpapi not installed, fallback to Google scraping")
        # Fallback to Google scraping
        try:
            from bs4 import BeautifulSoup
            import urllib.parse
            search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
            }
            resp = requests.get(search_url, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            search_results = soup.select('a[href^="http"]')
            count = 0
            for link in search_results:
                href = link.get('href')
                logging.info(f"Found link: {href}")
                if href and "google.com" not in href and count < top_k:
                    try:
                        page_resp = requests.get(href, timeout=10)
                        page_resp.raise_for_status()
                        page_text = strip_tags(page_resp.text)
                        results.append({"text": page_text[:800], "source": href, "page": None})
                        count += 1
                    except Exception as e:
                        logging.error(f"Error fetching page {href}: {e}")
                        continue
        except Exception as e:
            logging.error(f"Google search scraping error: {e}")
    except Exception as e:
        logging.error(f"SerpAPI search error: {e}")

    logging.info(f"search_web results count: {len(results)}")
    return results

# -------------------- Summarise with Gemini --------------------
def synthesize_information(chunks, objective):
    full_text="\n\n".join([c['text'] for c in chunks])
    model=genai.GenerativeModel("gemini-1.5-flash")
    # Updated prompt to request a longer detailed summary of 20-40 lines
    prompt=f"Combine and summarise the following text chunks in a detailed manner (20-40 lines) to answer: {objective}\n\n{full_text}"
    resp=model.generate_content(prompt)
    return resp.text

# -------------------- Embedding and FAISS Setup --------------------
embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

def embed_texts(texts):
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    return embeddings

def build_faiss_index(chunks):
    texts = [chunk['text'] for chunk in chunks]
    embeddings = embed_texts(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, chunks

def search_faiss_index(index, chunks, query, top_k=5):
    query_embedding = embed_texts([query])
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for idx in indices[0]:
        if idx < len(chunks):
            results.append(chunks[idx])
    return results

# -------------------- Streamlit UI --------------------
st.title("📚 Research Agent")

pdf_file=st.file_uploader("Upload a PDF (optional)", type="pdf")
question=st.text_input("Ask a question")
if question.strip() != "":
    with st.spinner("Searching..."):
        if pdf_file is not None:
            pdf_chunks = search_documents(question, pdf_file)
            if not pdf_chunks:
                st.warning("No relevant info found in the PDF.")
            else:
                index, chunks = build_faiss_index(pdf_chunks)
                relevant_chunks = search_faiss_index(index, chunks, question, top_k=5)
                answer = synthesize_information(relevant_chunks, question)
                st.subheader("Answer")
                st.text_area("", answer, height=400)
                st.subheader("Sources")
                # Collect unique sources with pages to avoid duplicates
                seen_sources = set()
                for c in relevant_chunks:
                    source_text = c['source'] if c['source'] else "Unknown source"
                    page_text = f" (page {c['page']})" if c['page'] else ""
                    source_display = f"[{source_text}]{page_text}"
                    if source_display not in seen_sources:
                        st.markdown(f"- {source_display}")
                        seen_sources.add(source_display)
        else:
            web_chunks = search_web(question)
            if not web_chunks:
                st.warning("No relevant info found on the web.")
            else:
                index, chunks = build_faiss_index(web_chunks)
                relevant_chunks = search_faiss_index(index, chunks, question, top_k=5)
                answer = synthesize_information(relevant_chunks, question)
                st.subheader("Answer")
                st.text_area("", answer, height=400)
                st.subheader("Sources")
                for c in relevant_chunks:
                    if c['page']:
                        st.markdown(f"- [{c['source']} (page {c['page']})]({c['source']})")
                    else:
                        st.markdown(f"- [{c['source']}]({c['source']})")
