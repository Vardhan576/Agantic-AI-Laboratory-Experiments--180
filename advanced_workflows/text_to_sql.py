import os
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google import genai

# Setup Gemini Client
# We check for a set environment variable or fall back to the project default key
api_key = os.environ.get("GEMINI_API_KEY") or "AQ.Ab8RN6KjqSW6dV56BW2aMsfa8dE8Yp8J9v1x7ooqUeUsqF1KOg"
client = genai.Client(api_key=api_key)

DB_PATH = os.path.join(os.path.dirname(__file__), "ecommerce.db")

# Define schema metadata for retrieval
# We associate each table's DDL with descriptions of the queries it can resolve.
schema_items = [
    {
        "table": "customers",
        "description": "Store customer demographic details, names, emails, countries of residence, and signup dates. Use this for queries about customer details, signups, geography, or lists of clients.",
        "ddl": """CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    country TEXT NOT NULL,
    signup_date TEXT NOT NULL
);"""
    },
    {
        "table": "products",
        "description": "Store product catalog details including product name, category, price, and stock levels. Use this for queries about product lists, prices, categories, inventory levels, or looking up product IDs.",
        "ddl": """CREATE TABLE products (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL,
    stock_quantity INTEGER NOT NULL
);"""
    },
    {
        "table": "orders",
        "description": "Store transaction details, orders, purchase date, quantities ordered, and total transaction amounts. Connects customers to products. Use this for queries about sales revenue, order history, purchase volume, quantity sold, or transactions.",
        "ddl": """CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    order_date TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    total_amount REAL NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);"""
    }
]

def build_schema_index():
    """Encode table descriptions and index them with FAISS."""
    print("[1/5] Encoding schemas and building FAISS index...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    texts_to_embed = [item["description"] for item in schema_items]
    embeddings = embedder.encode(texts_to_embed)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    return embedder, index

def retrieve_relevant_schemas(query, embedder, index, k=2):
    """Retrieve top-k relevant tables based on semantic similarity of query."""
    print(f"[2/5] Retrieving top {k} relevant tables for query: '{query}'...")
    query_embedding = embedder.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, k=k)
    
    retrieved = []
    print("   Retrieved Tables:")
    for idx in indices[0]:
        item = schema_items[idx]
        retrieved.append(item)
        print(f"    - {item['table']} (Description Match)")
        
    return retrieved

def generate_sql(query, retrieved_schemas):
    """Ask Gemini to generate the correct SQL query."""
    print("[3/5] Requesting SQL generation from Gemini...")
    
    schema_context = "\n\n".join([f"Table: {item['table']}\nDDL:\n{item['ddl']}" for item in retrieved_schemas])
    
    prompt = f"""
You are an expert SQLite developer. Given the database schema below and a user question, write a valid SQLite query to answer the question.
Do NOT include any markdown formatting, backticks, or explanation in your response. Return ONLY the raw SQL query.

Database Schemas:
{schema_context}

User Question: {query}

SQL Query:
"""
    
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    
    sql = response.text.strip()
    # Clean output in case the LLM returned markdown blocks
    if sql.startswith("```sql"):
        sql = sql[6:]
    if sql.endswith("```"):
        sql = sql[:-3]
    sql = sql.strip()
    
    print(f"   Generated SQL Query:\n   {sql}")
    return sql

def execute_query(sql):
    """Execute the SQL query on the SQLite database."""
    print("[4/5] Executing query on database...")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        
        print(f"   Returned {len(rows)} row(s).")
        return columns, rows, None
    except Exception as e:
        print(f"   Query Execution Error: {e}")
        return None, None, str(e)

def generate_final_response(query, sql, columns, rows, error):
    """Use Gemini to summarize the query output into a natural response."""
    print("[5/5] Synthesizing final natural language response...")
    
    if error:
        prompt = f"""
The user asked: "{query}"
We generated the SQL: {sql}
However, the database returned an error: {error}
Explain what went wrong in a user-friendly way.
"""
    else:
        results_str = f"Columns: {columns}\nRows:\n" + "\n".join([str(row) for row in rows])
        prompt = f"""
You are a helpful customer support and data analysis assistant. Given the user question, the SQL query executed, and the query results, synthesize a friendly, clear, and direct natural language answer for the user. Do not explain the SQL query itself, just answer the question based on the results.

User Question: {query}
SQL Query: {sql}
SQL Results:
{results_str}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    return response.text.strip()

def run_workflow(user_query):
    print("=" * 60)
    print(f"Starting Text-to-SQL Workflow for: '{user_query}'")
    print("=" * 60)
    
    embedder, index = build_schema_index()
    retrieved = retrieve_relevant_schemas(user_query, embedder, index, k=2)
    sql = generate_sql(user_query, retrieved)
    columns, rows, error = execute_query(sql)
    
    answer = generate_final_response(user_query, sql, columns, rows, error)
    
    print("\n" + "=" * 30 + " ANSWER " + "=" * 30)
    print(answer)
    print("=" * 68 + "\n")

if __name__ == "__main__":
    # Test query
    user_query = input("Ask a question about the ecommerce data (or press enter for default): ")
    if not user_query.strip():
        user_query = "What is the total revenue of products bought by Alice Smith?"
    run_workflow(user_query)
