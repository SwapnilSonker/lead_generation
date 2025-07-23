import os
import time
import json
import csv
import argparse
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from typing import TypedDict, List
import google.generativeai as genai
from serpapi import GoogleSearch
from langgraph.graph import StateGraph, END

# --- SETUP: Load API Keys & Hardcoded Configuration ---
load_dotenv()

# Configuration is now hardcoded in the script
SETTINGS = {
  "leads_to_find": 5,
  "api_timeout_seconds": 15,
  "rate_limit_delay_seconds": 20
}

PROMPTS = {
    "insight_extraction": "You are a business analyst. Read the following text from a company's website and summarize their core business in 2-3 concise bullet points. Focus on what they sell and who their customers are. Output ONLY the bullet points.",
    "lead_scoring": "You are a sales development representative for a computer hardware store. Read the following business insights. On a scale of 1-10, how strong of a fit is this company for our services (high-performance workstations, servers), where 10 is a perfect fit? Look for signals like company growth, complex operations, or a large technical team. Return ONLY the number and a one-sentence justification. Example: '8/10: Their focus on AI-driven logistics suggests a need for powerful processing hardware.'",
    "message_generation": "You are a B2B sales expert for a computer hardware store. Your goal is to write a personalized outreach email. Use the following key points and lead score to make your message highly relevant and specific. Reference one of the key points directly. Conclude with a clear call to action."
}

# --- HELPER FUNCTION for Deduplication ---
def normalize_url(url):
    """Strips scheme and 'www.' for accurate duplicate checking."""
    parsed = urlparse(url)
    return parsed.netloc.replace('www.', '') + parsed.path.rstrip('/')

# --- LANGGRAPH STATE DEFINITION ---
class GraphState(TypedDict):
    search_criteria: dict
    listicle_urls: List[str]
    company_names: List[str]
    final_leads: List[dict]

# --- PIPELINE NODES ---

def find_listicle_pages_node(state):
    print("--- 🔍 STEP 1: FINDING SOURCE ARTICLES ---")
    criteria = state['search_criteria']
    query = f"top {criteria['industry']} companies of size {criteria['size']} in {criteria.get('location', '')}"
    search = GoogleSearch({"q": query, "api_key": os.getenv("SERPAPI_API_KEY")})
    results = search.get_dict().get("organic_results", [])
    urls = [res["link"] for res in results[:3]]
    print(f"  > Found {len(urls)} articles to research.")
    return {"listicle_urls": urls}

def extract_company_names_node(state):
    print("\n--- 🤖 STEP 2: READING ARTICLES & EXTRACTING COMPANY NAMES ---")
    urls = state['listicle_urls']
    all_names = set()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    for url in urls:
        try:
            response = requests.get(url, timeout=SETTINGS['api_timeout_seconds'], headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.body.get_text(separator=' ', strip=True)[:4000]
            prompt = f"Read the following text. Identify specific company names mentioned. Return ONLY a comma-separated list of names.\n\nARTICLE TEXT:\n{text}"
            
            print(f"  > Analyzing {url}...")
            api_response = model.generate_content(prompt)
            names = [name.strip() for name in api_response.text.split(',') if len(name.strip()) > 2]
            all_names.update(names)
            time.sleep(SETTINGS['rate_limit_delay_seconds'])
        except requests.exceptions.RequestException as e:
            print(f"  > Failed to process {url}: {e}")
        except AttributeError:
             print(f"  > Failed to parse body for {url}. Page might lack a standard body tag.")


    print(f"\n  > Found {len(all_names)} unique potential company names.")
    return {"company_names": list(all_names)[:15]}

def find_and_deduplicate_sites_node(state):
    print("\n--- 🌐 STEP 3: FINDING & DEDUPLICATING OFFICIAL WEBSITES ---")
    names = state['company_names']
    clean_leads = []
    processed_urls = set()

    for name in names:
        try:
            query = f"{name} official website"
            search = GoogleSearch({"q": query, "api_key": os.getenv("SERPAPI_API_KEY")})
            result = search.get_dict().get("organic_results", [])
            
            if result:
                url = result[0]['link']
                normalized = normalize_url(url)
                if normalized not in processed_urls:
                    processed_urls.add(normalized)
                    clean_leads.append({"name": name, "website": url})
                    print(f"  > Found: {name} ({url})")
                else:
                    print(f"  > Skipping duplicate: {name} ({url})")
            time.sleep(1)
        except Exception as e:
            print(f"  > Error finding site for {name}: {e}")
        
        if len(clean_leads) >= SETTINGS['leads_to_find']:
            break
            
    return {"final_leads": clean_leads}

def process_leads_node(state):
    print("\n--- ✍️ STEP 4: SCRAPING, SCORING, & GENERATING MESSAGES ---")
    leads = state['final_leads']
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-2.5-flash')

    for lead in leads:
        try:
            print(f"\n  > Processing lead: {lead['name']}")
            response = requests.get(lead['website'], timeout=SETTINGS['api_timeout_seconds'], headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # --- ERROR FIX ---
            # Check if the body tag exists before trying to access it
            if not soup.body:
                print(f"    > FAILED: Page for {lead['name']} has no <body> tag. Skipping.")
                lead['scraped_insights'] = "ERROR: No <body> tag found on page."
                lead['lead_score'] = "ERROR"
                lead['personalized_message'] = "ERROR"
                continue # Move to the next lead

            text = soup.body.get_text(separator=' ', strip=True)[:3000]

            print("    > Extracting key insights...")
            insight_response = model.generate_content(f"{PROMPTS['insight_extraction']}\n\nWEBSITE TEXT: '{text}'")
            lead['scraped_insights'] = insight_response.text
            time.sleep(SETTINGS['rate_limit_delay_seconds'])

            print("    > Scoring lead...")
            score_response = model.generate_content(f"{PROMPTS['lead_scoring']}\n\nBUSINESS INSIGHTS:\n{lead['scraped_insights']}")
            lead['lead_score'] = score_response.text
            time.sleep(SETTINGS['rate_limit_delay_seconds'])

            print("    > Generating personalized message...")
            message_prompt = f"{PROMPTS['message_generation']}\n\nCOMPANY NAME: {lead['name']}\nKEY INSIGHTS:\n{lead['scraped_insights']}\nLEAD SCORE: {lead['lead_score']}\n\nYOUR PERSONALIZED EMAIL:"
            message_response = model.generate_content(message_prompt)
            lead['personalized_message'] = message_response.text
            time.sleep(SETTINGS['rate_limit_delay_seconds'])

        except Exception as e:
            print(f"    > FAILED to process lead {lead['name']}: {e}")
            lead['scraped_insights'] = "ERROR"
            lead['lead_score'] = "ERROR"
            lead['personalized_message'] = "ERROR"

    return {"final_leads": leads}

# --- BUILD AND COMPILE THE LANGGRAPH ---
workflow = StateGraph(GraphState)
workflow.add_node("find_articles", find_listicle_pages_node)
workflow.add_node("extract_names", extract_company_names_node)
workflow.add_node("find_sites", find_and_deduplicate_sites_node)
workflow.add_node("process_leads", process_leads_node)
workflow.set_entry_point("find_articles")
workflow.add_edge("find_articles", "extract_names")
workflow.add_edge("extract_names", "find_sites")
workflow.add_edge("find_sites", "process_leads")
workflow.add_edge("process_leads", END)
app = workflow.compile()

# --- EXECUTE THE PIPELINE with INTERACTIVE INPUT ---
if __name__ == "__main__":
    print("--- Lead Generation Script ---")
    print("Please provide the following details to start the search.")

    # Interactive prompts for user input
    industry_input = input("Enter the target industry (e.g., 'cybersecurity'): ")
    size_input = input("Enter the company size range (e.g., '50-200 employees'): ")
    location_input = input("Enter the target location (e.g., 'San Francisco'): ")
    
    # Use default values if input is empty
    search_criteria = {
        "industry": industry_input or "cybersecurity",
        "size": size_input or "50-200 employees",
        "location": location_input or "San Francisco"
    }

    print("\nStarting lead generation with the following criteria:")
    print(f"  Industry: {search_criteria['industry']}")
    print(f"  Size: {search_criteria['size']}")
    print(f"  Location: {search_criteria['location']}\n")
    
    inputs = {"search_criteria": search_criteria}
    final_state = app.invoke(inputs)
    final_leads = final_state.get('final_leads', [])
    
    # --- SAVE OUTPUT TO FILES ---
    if final_leads:
        # Save to JSON
        with open('leads_output.json', 'w', encoding='utf-8') as f:
            json.dump(final_leads, f, indent=4)
        print("\n\n✅ Results saved to leads_output.json")

        # Save to CSV
        fieldnames = final_leads[0].keys()
        with open('leads_output.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_leads)
        print("✅ Results saved to leads_output.csv")

    # --- PRINT FINAL DEMO OUTPUT ---
    print("\n\n--- 🏁 FINAL DEMO OUTPUT ---")
    for i, lead in enumerate(final_leads, 1):
        print(f"\n--- Lead {i} ---")
        print(f"Company Name:       {lead.get('name')}")
        print(f"Website:            {lead.get('website')}")
        print(f"Lead Score:         {lead.get('lead_score', 'N/A').strip()}")
        print("\nScraped Insights:")
        insights = lead.get('scraped_insights', 'N/A')
        for line in insights.split('\n'):
            if line.strip(): print(f"  {line.strip()}")
        print("\nPersonalized Message:")
        print(lead.get('personalized_message', 'N/A'))
        print("-" * 25)