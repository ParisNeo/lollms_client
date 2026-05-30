import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def generate_sovereign_dataset():
    # Set seed for reproducible deterministic generation
    np.random.seed(42)
    
    # ── 1. Create Target Directory ──
    dest_dir = Path("./data_workspace")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating synthetic relational dataset...")
    
    # ── 2. Customers Table (1,000 rows) ──
    num_customers = 1000
    customer_ids = [f"CUST-{i:04d}" for i in range(1, num_customers + 1)]
    
    first_names = ["Emma", "Liam", "Olivia", "Noah", "Ava", "Oliver", "Sophia", "Elijah", "Isabella", "James", "Mia", "Benjamin"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez"]
    countries = ["United States", "France", "Germany", "United Kingdom", "Canada", "Japan", "Australia"]
    tiers = ["Bronze", "Silver", "Gold", "Platinum"]
    tier_probs = [0.55, 0.25, 0.15, 0.05]
    
    cust_data = {
        "customer_id": customer_ids,
        "name": [f"{np.random.choice(first_names)} {np.random.choice(last_names)}" for _ in range(num_customers)],
        "country": np.random.choice(countries, size=num_customers),
        "signup_date": [
            (datetime(2024, 1, 1) + timedelta(days=int(np.random.randint(0, 365)))).strftime("%Y-%m-%d")
            for _ in range(num_customers)
        ],
        "membership_tier": np.random.choice(tiers, size=num_customers, p=tier_probs)
    }
    df_customers = pd.DataFrame(cust_data)
    
    # ── 3. Products Table (50 rows) ──
    num_products = 50
    product_ids = [f"PROD-{i:02d}" for i in range(1, num_products + 1)]
    categories = ["Electronics", "Apparel", "Home & Living", "Beauty & Care", "Sports & Outdoors"]
    
    prod_names = {
        "Electronics": ["Smartphone Alpha", "Wireless Earbuds", "Bluetooth Speaker", "Smart Watch v2", "USB-C Hub", "Tablet Pro", "Mechanical Keyboard"],
        "Apparel": ["Organic Cotton Tee", "Slim Fit Denim", "Wool Blend Sweater", "Waterproof Shell Jacket", "Athletic Socks Pack", "Leather Belt"],
        "Home & Living": ["Ergonomic Desk Chair", "LED Desk Lamp", "Ceramic Mug Set", "Memory Foam Pillow", "Scented Soy Candle", "Silicon Baking Mats"],
        "Beauty & Care": ["Moisturizing Cream", "Sandalwood Beard Oil", "Hydrating Lip Balm", "Bamboo Toothbrush Pack", "Mineral Sunscreen"],
        "Sports & Outdoors": ["Vacuum Water Bottle", "Lightweight Yoga Mat", "Resistance Bands Set", "Adjustable Dumbbells", "Hiking Backpack"]
    }
    
    products_list = []
    for i, pid in enumerate(product_ids):
        cat = np.random.choice(categories)
        pname = f"{np.random.choice(prod_names[cat])} {np.random.choice(['Classic', 'Elite', 'Eco', 'Special Edition', 'Lite'])}"
        
        # Calculate pricing and Cost of Goods Sold (COGS)
        price = round(float(np.random.uniform(9.99, 299.99)), 2)
        # COGS is between 35% and 65% of the sales price
        cogs = round(price * float(np.random.uniform(0.35, 0.65)), 2)
        stock = int(np.random.randint(5, 500))
        
        products_list.append({
            "product_id": pid,
            "product_name": pname,
            "category": cat,
            "price": price,
            "cost_of_goods_sold": cogs,
            "stock_qty": stock
        })
    df_products = pd.DataFrame(products_list)
    
    # ── 4. Orders Table (2,500 rows) ──
    num_orders = 2500
    order_ids = [f"ORD-{i:05d}" for i in range(1, num_orders + 1)]
    payment_methods = ["Credit Card", "PayPal", "Crypto", "Bank Transfer"]
    
    start_date = datetime(2025, 1, 1)
    orders_list = []
    for oid in order_ids:
        cust_id = np.random.choice(customer_ids)
        # Spread orders over the 2025 calendar year
        days_offset = int(np.random.randint(0, 365))
        order_date = (start_date + timedelta(days=days_offset)).strftime("%Y-%m-%d")
        pay_method = np.random.choice(payment_methods, p=[0.50, 0.30, 0.10, 0.10])
        
        orders_list.append({
            "order_id": oid,
            "customer_id": cust_id,
            "order_date": order_date,
            "payment_method": pay_method
        })
    df_orders = pd.DataFrame(orders_list)
    
    # ── 5. Order Details Table (~6,000 rows) ──
    order_details_list = []
    detail_counter = 1
    
    for oid in order_ids:
        # Each order contains between 1 and 5 unique items
        num_items = int(np.random.randint(1, 6))
        chosen_prods = np.random.choice(product_ids, size=num_items, replace=False)
        
        for pid in chosen_prods:
            qty = int(np.random.choice([1, 2, 3, 5], p=[0.70, 0.15, 0.10, 0.05]))
            # 20% chance of receiving a discount (either 10%, 15%, or 20% off)
            has_discount = np.random.rand() < 0.20
            discount = float(np.random.choice([0.10, 0.15, 0.20])) if has_discount else 0.0
            
            order_details_list.append({
                "order_detail_id": f"DET-{detail_counter:06d}",
                "order_id": oid,
                "product_id": pid,
                "quantity": qty,
                "discount": discount
            })
            detail_counter += 1
    df_details = pd.DataFrame(order_details_list)
    
    # ── 6. Save Relational Spreadsheet (.xlsx) ──
    xlsx_path = dest_dir / "sales_database.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_customers.to_excel(writer, sheet_name="Customers", index=False)
        df_products.to_excel(writer, sheet_name="Products", index=False)
        df_orders.to_excel(writer, sheet_name="Orders", index=False)
        df_details.to_excel(writer, sheet_name="Order_Details", index=False)
    print(f"Relational spreadsheet generated at: {xlsx_path.resolve()}")
    
    # ── 7. Save Denormalized Flat File (.csv) ──
    # Merges all tables into a single large flat sales table for flat-file testing
    df_flat = df_details.merge(df_orders, on="order_id") \
                        .merge(df_customers, on="customer_id") \
                        .merge(df_products, on="product_id")
                        
    # Calculate derived financial columns for testing calculations
    df_flat["gross_revenue"] = df_flat["quantity"] * df_flat["price"]
    df_flat["total_discount"] = df_flat["gross_revenue"] * df_flat["discount"]
    df_flat["net_revenue"] = df_flat["gross_revenue"] - df_flat["total_discount"]
    df_flat["total_cost"] = df_flat["quantity"] * df_flat["cost_of_goods_sold"]
    df_flat["net_profit"] = df_flat["net_revenue"] - df_flat["total_cost"]
    
    csv_path = dest_dir / "flat_sales_data.csv"
    df_flat.to_csv(csv_path, index=False)
    print(f"Denormalized flat sales file generated at: {csv_path.resolve()}")
    print("Dataset generation successful.")

if __name__ == "__main__":
    generate_sovereign_dataset()
