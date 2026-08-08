import sqlite3
import os

def init_db():
    db_path = os.path.join(os.path.dirname(__file__), "ecommerce.db")
    print(f"Initializing database at: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        country TEXT NOT NULL,
        signup_date TEXT NOT NULL
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        product_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        price REAL NOT NULL,
        stock_quantity INTEGER NOT NULL
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        order_id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id INTEGER NOT NULL,
        product_id INTEGER NOT NULL,
        order_date TEXT NOT NULL,
        quantity INTEGER NOT NULL,
        total_amount REAL NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    )
    """)
    
    # Insert mock data if tables are empty
    cursor.execute("SELECT COUNT(*) FROM customers")
    if cursor.fetchone()[0] == 0:
        customers = [
            ("Alice Smith", "alice@example.com", "USA", "2026-01-15"),
            ("Bob Jones", "bob@example.com", "Canada", "2026-02-20"),
            ("Charlie Brown", "charlie@example.com", "UK", "2026-03-05"),
            ("David Lee", "david@example.com", "Australia", "2026-04-10"),
            ("Emma Watson", "emma@example.com", "France", "2026-05-12"),
        ]
        cursor.executemany("INSERT INTO customers (name, email, country, signup_date) VALUES (?, ?, ?, ?)", customers)
        print(f"Inserted {len(customers)} customers.")

    cursor.execute("SELECT COUNT(*) FROM products")
    if cursor.fetchone()[0] == 0:
        products = [
            ("Laptop Pro", "Electronics", 1200.00, 15),
            ("Smartphone X", "Electronics", 800.00, 30),
            ("Wireless Headphone", "Accessories", 150.00, 50),
            ("Mechanical Keyboard", "Accessories", 100.00, 40),
            ("Coffee Maker", "Home Appliances", 90.00, 25),
            ("Desk Lamp", "Home Appliances", 35.00, 60),
        ]
        cursor.executemany("INSERT INTO products (name, category, price, stock_quantity) VALUES (?, ?, ?, ?)", products)
        print(f"Inserted {len(products)} products.")

    cursor.execute("SELECT COUNT(*) FROM orders")
    if cursor.fetchone()[0] == 0:
        orders = [
            (1, 1, "2026-06-01", 1, 1200.00), # Alice bought Laptop Pro
            (1, 3, "2026-06-15", 2, 300.00),  # Alice bought 2 Wireless Headphones
            (2, 2, "2026-06-10", 1, 800.00),  # Bob bought Smartphone X
            (3, 4, "2026-07-02", 1, 100.00),  # Charlie bought Keyboard
            (4, 5, "2026-07-11", 1, 90.00),   # David bought Coffee Maker
            (5, 6, "2026-07-20", 3, 105.00),  # Emma bought 3 Desk Lamps
            (2, 3, "2026-08-01", 1, 150.00),  # Bob bought Wireless Headphone
            (3, 1, "2026-08-05", 1, 1200.00), # Charlie bought Laptop Pro
        ]
        cursor.executemany("INSERT INTO orders (customer_id, product_id, order_date, quantity, total_amount) VALUES (?, ?, ?, ?, ?)", orders)
        print(f"Inserted {len(orders)} orders.")
        
    conn.commit()
    conn.close()
    print("Database setup complete.")

if __name__ == "__main__":
    init_db()
