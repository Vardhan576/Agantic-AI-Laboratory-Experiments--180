import os
import json
from google import genai

# Setup Gemini Client
API_KEY = os.environ.get("GEMINI_API_KEY") or "AQ.Ab8RN6KjqSW6dV56BW2aMsfa8dE8Yp8J9v1x7ooqUeUsqF1KOg"
client = None
if os.environ.get("GEMINI_API_KEY") or API_KEY != "AQ.Ab8RN6KjqSW6dV56BW2aMsfa8dE8Yp8J9v1x7ooqUeUsqF1KOg":
    try:
        client = genai.Client(api_key=API_KEY)
    except Exception:
        client = None

MODEL_NAME = "gemini-3.5-flash"

def lead_generation_agent(target_segment: str, use_mock=False) -> list:
    """Agent 1: Generates target leads based on a market segment."""
    print(f"[Lead Gen Agent] Generating leads for segment: '{target_segment}'...")
    
    # Pre-coded high-fidelity mock data
    mock_leads = [
        {
            "company_name": "Acme Talent Solutions",
            "industry": "Recruitment & HR Tech",
            "company_size": 120,
            "contact_name": "Sarah Jenkins",
            "contact_email": "sarah.jenkins@acmetalent.com",
            "company_description": "Provides outsourced recruitment services to mid-market tech firms.",
            "pain_point": "Struggling to scale candidate matching processes, leading to long placement cycles."
        },
        {
            "company_name": "LogiCorp Global",
            "industry": "Supply Chain & Logistics",
            "company_size": 450,
            "contact_name": "Marcus Vance",
            "contact_email": "m.vance@logicorp.com",
            "company_description": "Manages international freight forwarding and warehousing solutions.",
            "pain_point": "Manual reporting of shipment statuses causes communication delays with premium clients."
        },
        {
            "company_name": "EduStream Academy",
            "industry": "EdTech",
            "company_size": 35,
            "contact_name": "Dr. Elena Rostova",
            "contact_email": "elena@edustream.org",
            "company_description": "Online learning platform specializing in K-12 interactive STEM curriculum.",
            "pain_point": "High student churn rates due to lack of personalized learning path recommendations."
        }
    ]
    
    if use_mock or not client:
        print("[Lead Gen Agent] [Mock] Returning simulated prospective leads.")
        return mock_leads
        
    prompt = f"""
    You are a Lead Generation Agent. Your goal is to identify and generate realistic prospective leads for a company operating in the target segment: '{target_segment}'.
    Generate a JSON list containing 3 realistic prospective lead companies. Each lead must contain:
    - company_name: Name of the company
    - industry: Industry type
    - company_size: Estimated number of employees (integer)
    - contact_name: Full name of the decision maker (e.g., CTO, HR Director)
    - contact_email: Email address
    - company_description: Brief summary of what they do
    - pain_point: A realistic business pain point they are likely facing

    Respond ONLY with the JSON array. Do not include markdown formatting or backticks.
    """
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        leads = json.loads(text)
        print(f"[Lead Gen Agent] Successfully generated {len(leads)} leads.")
        return leads
    except Exception as e:
        print(f"[Lead Gen Agent] Warning: API Error ({e}). Falling back to simulation mode.")
        return mock_leads

def qualification_agent(lead: dict, criteria: str, use_mock=False) -> dict:
    """Agent 2: Evaluates a lead against Ideal Customer Profile (ICP) criteria."""
    company = lead["company_name"]
    print(f"[Qualification Agent] Evaluating lead: {company}...")
    
    if use_mock or not client:
        # Evaluate mock score
        size = lead["company_size"]
        if size >= 50:
            score = 85
            status = "QUALIFIED"
            reasons = ["Company size meets ICP minimum of 50.", "Stated pain point matches core automation capabilities."]
        else:
            score = 45
            status = "DISQUALIFIED"
            reasons = ["Company size is below the 50 employees threshold.", "Higher churn risk for micro-businesses."]
            
        lead.update({
            "qualification_score": score,
            "status": status,
            "reasons": reasons
        })
        print(f"[Qualification Agent] [Mock] {company} scored {score} ({status})")
        return lead
        
    prompt = f"""
    You are a Sales Qualification Agent. Compare the prospective lead details below against the Ideal Customer Profile (ICP) criteria.
    
    Lead Details:
    {json.dumps(lead, indent=2)}
    
    ICP Criteria:
    {criteria}
    
    Generate a JSON response containing:
    - qualification_score: A numerical score between 0 and 100 representing how well they match.
    - status: Either 'QUALIFIED' (score >= 60) or 'DISQUALIFIED' (score < 60)
    - reasons: A list of specific reasons explaining the score based on company size, fit, and pain point.

    Respond ONLY with the JSON. Do not include markdown formatting or backticks.
    """
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        evaluation = json.loads(text)
        lead.update(evaluation)
        print(f"[Qualification Agent] {company} scored {evaluation.get('qualification_score')} ({evaluation.get('status')})")
        return lead
    except Exception as e:
        print(f"[Qualification Agent] Warning: API Error ({e}). Falling back to simulation mode.")
        score = 80 if lead["company_size"] >= 50 else 45
        status = "QUALIFIED" if score >= 60 else "DISQUALIFIED"
        lead.update({
            "qualification_score": score,
            "status": status,
            "reasons": [f"Company size is {lead['company_size']}", "Pain point aligns with technical capabilities."]
        })
        return lead

def email_copywriter_agent(lead: dict, product_value_prop: str, use_mock=False) -> str:
    """Agent 3: Generates a personalized sales email for qualified leads."""
    company = lead["company_name"]
    if lead["status"] != "QUALIFIED":
        print(f"[Email Agent] Skipping {company} (DISQUALIFIED).")
        return ""
        
    print(f"[Email Agent] Generating personalized email for {lead['contact_name']} at {company}...")
    
    if use_mock or not client:
        subject = f"Improving operational scalability at {company}"
        body = f"Hi {lead['contact_name']},\n\nI noticed that {company} is currently expanding its footprint in the {lead['industry']} sector. Given your focus on operational efficiency, I wanted to reach out regarding your current process bottlenecks around {lead['pain_point'].lower()}.\n\nOur agentic AI workflows are built precisely to automate these manual workflows, reducing cycle times by up to 80%.\n\nWould you be open to a brief, 10-minute introductory call next Tuesday to discuss how we might help?\n\nBest regards,\nAPEX Operations Team"
        print("[Email Agent] [Mock] Generated template outreach email.")
        return f"Subject: {subject}\n\n{body}"
        
    prompt = f"""
    You are an Email Copywriter Agent. Write a highly personalized, compelling B2B cold outreach email to the contact person.
    
    Lead Details:
    - Company: {lead['company_name']}
    - Contact: {lead['contact_name']}
    - Email: {lead['contact_email']}
    - Pain Point: {lead['pain_point']}
    - Business Context: {lead['company_description']}
    
    Our Product Value Proposition:
    {product_value_prop}
    
    Constraints:
    - Subject line should be punchy and personalized (no emojis).
    - Email body should be brief, conversational, and direct (under 150 words).
    - End with a low-friction call-to-action (e.g., asking for a short chat next week).
    - Do not use corporate jargon or buzzwords.
    - Format response as:
      Subject: [Subject Line]
      
      [Email Body]
    """
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        email = response.text.strip()
        print(f"[Email Agent] Email generated successfully.")
        return email
    except Exception as e:
        print(f"[Email Agent] Warning: API Error ({e}). Falling back to simulation template.")
        subject = f"Optimizing {lead['industry']} workflows at {company}"
        body = f"Hi {lead['contact_name']},\n\nI wanted to reach out since our platform helps resolve {lead['pain_point'].lower()} by automating core operational steps.\n\nLet's schedule a 10-minute call next week to see if we can help {company} save up to 80% on manual task times.\n\nBest,\nSDR Team"
        return f"Subject: {subject}\n\n{body}"

def main():
    print("="*60)
    print("EXPERIMENT 5: MULTI-AGENT SDR SYSTEM START")
    print("="*60)
    
    use_mock = False
    if API_KEY == "AQ.Ab8RN6KjqSW6dV56BW2aMsfa8dE8Yp8J9v1x7ooqUeUsqF1KOg" and not os.environ.get("GEMINI_API_KEY"):
        use_mock = True
        print("[System Info] Gemini API Key not set. Running in local high-fidelity simulation mode.")
    
    # Define target market and ICP
    target_segment = "Mid-sized and enterprise firms needing automation/AI in operations"
    product_value_prop = "We provide agentic AI workflows that automate routine operations, database queries, and report generation, reducing manual work hours by 80%."
    icp_criteria = """
    - B2B company operating in Tech, Logistics, HR, or EdTech.
    - Company size of at least 50 employees (mid-market or enterprise).
    - Active pain points involving manual bottlenecks, long process cycles, or scaling issues.
    """
    
    # 1. Lead Gen
    raw_leads = lead_generation_agent(target_segment, use_mock=use_mock)
    
    # 2. Qualification
    evaluated_leads = []
    for lead in raw_leads:
        evaluated_lead = qualification_agent(lead, icp_criteria, use_mock=use_mock)
        evaluated_leads.append(evaluated_lead)
        
    # 3. Email Writing & Output Compilation
    results = []
    print("\n" + "="*50 + "\nOUTREACH CAMPAIGN RESULTS\n" + "="*50)
    for lead in evaluated_leads:
        email_text = email_copywriter_agent(lead, product_value_prop, use_mock=use_mock)
        lead_result = {
            "lead_info": lead,
            "outreach_email": email_text
        }
        results.append(lead_result)
        
        print(f"\nCompany: {lead['company_name']} ({lead['industry']})")
        print(f"Size: {lead['company_size']} employees")
        print(f"Score: {lead['qualification_score']} -> {lead['status']}")
        print(f"Reasons: {', '.join(lead['reasons'])}")
        if email_text:
            print("-" * 40)
            print(email_text)
            print("-" * 40)
        else:
            print("Action: No outreach sent.")
            
    # Save reports
    out_dir = os.path.dirname(__file__)
    report_path = os.path.join(out_dir, "sdr_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n[SDR System] Campaign report saved to {report_path}")
    print("="*60)

if __name__ == "__main__":
    main()
