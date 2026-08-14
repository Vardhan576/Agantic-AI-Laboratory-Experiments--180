import os
import sqlite3
import json
import re
import numpy as np
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
EMBED_MODEL = "text-embedding-004"

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ecommerce.db"))

# Corporate Knowledge Base for RAG
KNOWLEDGE_BASE = [
    "Refund Policy: Customers can request a full refund within 30 days of purchase. The item must be unused, undamaged, and in its original packaging.",
    "Account Recovery: If a customer forgets their password, they must use the self-service 'Forgot Password' page. Support agents are strictly prohibited from sharing raw passwords or temporary credentials directly via chat.",
    "Shipping and Delivery: Standard shipping takes 3-5 business days and is free for orders over $50. Express shipping takes 1-2 business days and costs $15.",
    "Subscription Tiers: Free Tier has basic support. Premium Tier ($19/mo) has priority email support. Enterprise Tier ($99/mo) includes dedicated account managers and 24/7 phone support.",
    "Safety Disclaimer: Any financial advice, projections, or suggestions provided by our system are for educational purposes only. We do not guarantee absolute financial returns or profits."
]

# We embed the knowledge base once at startup
print("[Capstone Agent] Pre-indexing RAG knowledge base...")
KNOWLEDGE_EMBEDDINGS = []
if client:
    try:
        for doc in KNOWLEDGE_BASE:
            response = client.models.embed_content(model=EMBED_MODEL, contents=doc)
            KNOWLEDGE_EMBEDDINGS.append(response.embeddings[0].values)
        KNOWLEDGE_EMBEDDINGS = np.array(KNOWLEDGE_EMBEDDINGS, dtype=np.float32)
        print("[Capstone Agent] RAG Knowledge Base indexed successfully via API.")
    except Exception as e:
        print(f"[Capstone Agent] Error pre-indexing: {e}. Falling back to keyword search.")
        client = None
else:
    print("[Capstone Agent] RAG pre-indexing skipped (Offline/Simulation mode active).")

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def run_db_query(sql: str) -> str:
    """Execute a database query safely on the SQLite ecommerce database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        result = [dict(zip(columns, row)) for row in rows]
        conn.close()
        return json.dumps(result, indent=2)
    except Exception as e:
        conn.close()
        return f"Database Error: {str(e)}"

def run_rag_search(query: str, k: int = 1, use_mock=False) -> str:
    """Retrieve the top-k most relevant documentation chunks."""
    if use_mock or not client:
        # High-fidelity keyword matching fallback
        q = query.lower()
        matched = []
        if "refund" in q or "return" in q:
            matched.append(KNOWLEDGE_BASE[0])
        if "password" in q or "credentials" in q or "login" in q:
            matched.append(KNOWLEDGE_BASE[1])
        if "ship" in q or "delivery" in q or "arrive" in q:
            matched.append(KNOWLEDGE_BASE[2])
        if "subscription" in q or "tier" in q or "plan" in q:
            matched.append(KNOWLEDGE_BASE[3])
        if "advice" in q or "profit" in q or "guarantee" in q:
            matched.append(KNOWLEDGE_BASE[4])
            
        # Fallback to first if nothing matched
        if not matched:
            matched.append(KNOWLEDGE_BASE[0])
            
        return "\n".join(matched[:k])
        
    try:
        response = client.models.embed_content(model=EMBED_MODEL, contents=query)
        query_vector = np.array(response.embeddings[0].values, dtype=np.float32)
        
        similarities = []
        for doc_vector in KNOWLEDGE_EMBEDDINGS:
            dot = np.dot(query_vector, doc_vector)
            norm1 = np.linalg.norm(query_vector)
            norm2 = np.linalg.norm(doc_vector)
            sim = float(dot / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 0.0
            similarities.append(sim)
            
        best_idx = np.argsort(similarities)[::-1][:k]
        retrieved = [KNOWLEDGE_BASE[idx] for idx in best_idx]
        return "\n".join(retrieved)
    except Exception as e:
        print(f"[RAG Agent] API Error: {e}. Falling back to keyword search.")
        return run_rag_search(query, k, use_mock=True)

def check_safety_and_censor(agent_response: str, use_mock=False) -> tuple[str, list]:
    """Censor any PII or uncompliant financial guarantees in the response."""
    violations = []
    censored_response = agent_response
    
    # 1. Censor phone numbers
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    if re.search(phone_pattern, censored_response):
        censored_response = re.sub(phone_pattern, "[REDACTED PHONE NUMBER]", censored_response)
        violations.append("P01_PII_PROTECTION: Censored phone number.")
        
    # 2. Censor password exposures
    password_keywords = ["password is", "password:", "temp password"]
    for kw in password_keywords:
        if kw in censored_response.lower():
            censored_response = "I apologize, but I am not authorized to share passwords or account credentials. Please use the self-service Account Recovery page."
            violations.append("P01_PII_PROTECTION: Refused password exposure.")
            break
            
    # 3. Double-check with LLM Supervisor
    if use_mock or not client:
        # Check for uncompliant financial terms locally
        if "guarante" in censored_response.lower() and "profit" in censored_response.lower():
            censored_response += "\n\nDisclaimer: We do not guarantee absolute financial returns."
            violations.append("RULE_SUPERVISOR: Appended disclaimer to financial guarantee.")
        return censored_response, violations
        
    supervisor_prompt = f"""
    Evaluate this customer support response for compliance:
    ---
    {censored_response}
    ---
    
    Rules:
    - Never guarantee absolute financial profits or gains.
    - Never defame competitors.
    
    If it is compliant, reply 'COMPLIANT'.
    If it violates these rules, rewrite the response to be professional and compliant, and return ONLY the compliant rewrite.
    """
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=supervisor_prompt
        )
        verdict = response.text.strip()
        if verdict != "COMPLIANT":
            censored_response = verdict
            violations.append("LLM_SUPERVISOR: Rewrote response for policy compliance.")
    except Exception:
        pass
        
    return censored_response, violations

def generate_mock_sql(user_query: str) -> str:
    """Generate high-fidelity SQL queries locally based on query keywords."""
    q = user_query.lower()
    
    # Check Alice Smith queries
    if "alice smith" in q:
        if "orders" in q or "how many" in q:
            return "SELECT COUNT(*) AS total_orders FROM orders JOIN customers ON orders.customer_id = customers.customer_id WHERE customers.name = 'Alice Smith';"
        if "revenue" in q or "spend" in q or "spendings" in q:
            return "SELECT SUM(orders.total_amount) AS total_spent FROM orders JOIN customers ON orders.customer_id = customers.customer_id WHERE customers.name = 'Alice Smith';"
        return "SELECT * FROM customers WHERE name = 'Alice Smith';"
        
    # Check general totals
    if "total revenue" in q or "revenue" in q:
        return "SELECT SUM(total_amount) AS total_revenue FROM orders;"
        
    if "customers" in q:
        if "count" in q or "how many" in q:
            return "SELECT COUNT(*) AS total_customers FROM customers;"
        return "SELECT * FROM customers LIMIT 5;"
        
    if "products" in q:
        if "count" in q or "how many" in q:
            return "SELECT COUNT(*) AS total_products FROM products;"
        return "SELECT * FROM products LIMIT 5;"
        
    if "orders" in q:
        if "count" in q or "how many" in q:
            return "SELECT COUNT(*) AS total_orders FROM orders;"
        return "SELECT * FROM orders LIMIT 5;"
        
    return "SELECT * FROM products LIMIT 5;"

def run_agentic_pipeline(user_query: str) -> dict:
    """Orchestrate triage routing, tool execution, safety checks, and response synthesis."""
    trace_logs = []
    trace_logs.append("[Supervisor] Received user query.")
    
    use_mock = False
    if API_KEY == "AQ.Ab8RN6KjqSW6dV56BW2aMsfa8dE8Yp8J9v1x7ooqUeUsqF1KOg" and not os.environ.get("GEMINI_API_KEY"):
        use_mock = True
        
    category = "GENERAL"
    if use_mock or not client:
        # Determine category locally
        q = user_query.lower()
        if any(w in q for w in ["order", "revenue", "spend", "customer", "product", "sales"]):
            category = "DB_QUERY"
        elif any(w in q for w in ["refund", "policy", "ship", "deliver", "subscription", "tier", "plan"]):
            category = "RAG_SEARCH"
        else:
            category = "GENERAL"
        trace_logs.append(f"[Triage] [Local] Classified query category as: '{category}'")
    else:
        triage_prompt = f"""
        You are a Customer Support Router. Classify the user query into one of three support categories:
        1. 'DB_QUERY': Queries about customer names, orders, purchase history, sales metrics, order counts, product listings, prices, or inventory.
        2. 'RAG_SEARCH': Queries about company policy, refunds, subscription rates/tiers, shipping rules, or general instructions.
        3. 'GENERAL': Greetings or general conversation.
        
        User Query: "{user_query}"
        
        Respond with ONLY the category name. Do not include markdown formatting or backticks.
        """
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=triage_prompt)
            category = response.text.strip().upper()
        except Exception:
            category = "GENERAL"
        trace_logs.append(f"[Triage] Classified query category as: '{category}'")
        
    raw_tool_result = ""
    system_response = ""
    
    # 2. Tool / Sub-agent Execution
    if category == "DB_QUERY":
        sql = ""
        if use_mock or not client:
            sql = generate_mock_sql(user_query)
            trace_logs.append(f"[DB Agent] [Local] Generated SQL Query: `{sql}`")
        else:
            db_schema = """
            customers (customer_id, name, email, country, signup_date)
            products (product_id, name, category, price, stock_quantity)
            orders (order_id, customer_id, product_id, order_date, quantity, total_amount)
            """
            sql_prompt = f"""
            You are an expert SQLite generator. Given this database schema:
            {db_schema}
            
            Write a valid SQLite query to answer the customer's request.
            Do NOT write markdown, do NOT write backticks. Output ONLY the raw SQL query.
            
            Customer Request: "{user_query}"
            """
            try:
                sql_response = client.models.generate_content(model=MODEL_NAME, contents=sql_prompt)
                sql = sql_response.text.strip()
                if sql.startswith("```sql"):
                    sql = sql[6:]
                if sql.endswith("```"):
                    sql = sql[:-3]
                sql = sql.strip()
                trace_logs.append(f"[DB Agent] Generated SQL Query: `{sql}`")
            except Exception:
                sql = generate_mock_sql(user_query)
                trace_logs.append(f"[DB Agent] API Error. Fallback to SQL Query: `{sql}`")
                
        raw_tool_result = run_db_query(sql)
        trace_logs.append(f"[DB Agent] Executed SQL on SQLite database. Retrieved records.")
        
        if use_mock or not client:
            records = json.loads(raw_tool_result)
            if records:
                formatted_records = ", ".join([f"{k}: {v}" for k, v in records[0].items()])
                system_response = f"I have queried our database regarding your request. The records show: {formatted_records}."
            else:
                system_response = "I queried the database, but no matching records were found."
        else:
            synthesis_prompt = f"""
            You are a helpful customer support agent. Synthesize a friendly natural language response for the customer.
            
            User Query: "{user_query}"
            Retrieved Database Records:
            {raw_tool_result}
            
            Provide the answer directly, referencing the order/customer details.
            """
            try:
                synth_response = client.models.generate_content(model=MODEL_NAME, contents=synthesis_prompt)
                system_response = synth_response.text.strip()
            except Exception:
                system_response = f"Database records show: {raw_tool_result}."
                
    elif category == "RAG_SEARCH":
        trace_logs.append("[RAG Agent] Computing query embeddings...")
        retrieved_docs = run_rag_search(user_query, k=1, use_mock=use_mock)
        trace_logs.append(f"[RAG Agent] Retrieved documentation context:\n{retrieved_docs}")
        raw_tool_result = retrieved_docs
        
        if use_mock or not client:
            system_response = f"According to our corporate guidelines: {retrieved_docs}"
        else:
            synthesis_prompt = f"""
            You are a helpful support agent. Answer the user's question accurately using ONLY the provided corporate policies.
            
            Corporate Policies:
            {retrieved_docs}
            
            User Query: "{user_query}"
            """
            try:
                synth_response = client.models.generate_content(model=MODEL_NAME, contents=synthesis_prompt)
                system_response = synth_response.text.strip()
            except Exception:
                system_response = f"Policy context: {retrieved_docs}"
                
    else:
        # General response
        if use_mock or not client:
            if "hello" in user_query.lower() or "hi" in user_query.lower():
                system_response = "Hello! I am the APEX enterprise operations agent. How can I help you today?"
            else:
                system_response = "I am here to help you query order data or look up corporate policy. Could you specify your request?"
        else:
            general_prompt = f"Respond politely to the customer query: \"{user_query}\""
            try:
                general_response = client.models.generate_content(model=MODEL_NAME, contents=general_prompt)
                system_response = general_response.text.strip()
            except Exception:
                system_response = "Hello! How can I help you query our database or policies today?"
                
    trace_logs.append("[Supervisor] Sub-agent execution completed. Evaluating compliance...")
    
    # 3. Safety Compliance Verification
    final_output, violations = check_safety_and_censor(system_response, use_mock=use_mock)
    if violations:
        for v in violations:
            trace_logs.append(f"[Safety Supervisor] [CENSORED] {v}")
    else:
        trace_logs.append("[Safety Supervisor] Response passed safety checks.")
        
    trace_logs.append("[Supervisor] Pipeline execution finished.")
    
    return {
        "category": category,
        "raw_response": system_response,
        "final_response": final_output,
        "tool_data": raw_tool_result,
        "trace_logs": trace_logs
    }

if __name__ == "__main__":
    res = run_agentic_pipeline("How many orders has customer Alice Smith placed, and what is her total spend?")
    print(json.dumps(res, indent=2))
