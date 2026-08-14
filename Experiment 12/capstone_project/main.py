import os
import sqlite3
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from capstone_agent import run_agentic_pipeline, DB_PATH

app = FastAPI(title="Capstone Enterprise Agentic Portal")

# Serve static files (CSS, JS)
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
os.makedirs(os.path.join(static_dir, "css"), exist_ok=True)
os.makedirs(os.path.join(static_dir, "js"), exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

class ChatRequest(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
def read_root():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="index.html template not found")
        
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
        
    try:
        result = run_agentic_pipeline(request.message)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
def status_endpoint():
    """Retrieve database metrics to show on the dashboard UI."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM customers;")
        customers_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM products;")
        products_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM orders;")
        orders_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(total_amount) FROM orders;")
        total_revenue = cursor.fetchone()[0]
        total_revenue = round(total_revenue or 0.0, 2)
        
        conn.close()
        return {
            "status": "online",
            "db_path": DB_PATH,
            "metrics": {
                "customers": customers_count,
                "products": products_count,
                "orders": orders_count,
                "revenue": total_revenue
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "db_path": DB_PATH,
            "error_msg": str(e),
            "metrics": {
                "customers": 0,
                "products": 0,
                "orders": 0,
                "revenue": 0.0
            }
        }
