import os
import sqlite3
from fastapi import FastAPI, Request, Response, Form
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="WRAITH Vulnerable Target")

# Setup a vulnerable SQLite in-memory DB
conn = sqlite3.connect(':memory:', check_same_thread=False)
cursor = conn.cursor()
cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, password TEXT, role TEXT)")
cursor.execute("INSERT INTO users (username, password, role) VALUES ('admin', 'supersecret', 'admin')")
cursor.execute("INSERT INTO users (username, password, role) VALUES ('alice', 'password123', 'user')")
conn.commit()

# Create dummy hidden files
with open(".env", "w") as f:
    f.write("DB_PASSWORD=production_secret_do_not_share\nAWS_KEY=AKIAIOSFODNN7EXAMPLE")

@app.get("/", response_class=HTMLResponse)
async def home():
    """Main entry point with links."""
    return """
    <html>
        <head><title>SecureCorp API</title></head>
        <body>
            <h1>Welcome to SecureCorp</h1>
            <p>Use our internal tools below:</p>
            <ul>
                <li><a href="/search">Search Portal</a></li>
                <li><a href="/api/users?id=2">User Lookup API</a></li>
                <li><a href="/login">Admin Login</a></li>
            </ul>
        </body>
    </html>
    """

@app.get("/search", response_class=HTMLResponse)
async def search(q: str = ""):
    """XSS Vulnerable Endpoint: Reflects 'q' without sanitization."""
    if q:
        # Intentionally vulnerable to Reflected XSS
        return f"<html><body><h1>Search Results</h1><p>You searched for: {q}</p></body></html>"
    return "<html><body><h1>Search Portal</h1><form action='/search'><input name='q'><button>Search</button></form></body></html>"

@app.get("/api/users")
async def get_user(id: str = "1"):
    """SQLi Vulnerable Endpoint: Concatenates 'id' directly into query."""
    try:
        # Intentionally vulnerable to SQL Injection
        query = f"SELECT id, username, role FROM users WHERE id = {id}"
        cursor.execute(query)
        result = cursor.fetchall()
        
        if result:
            users = [{"id": r[0], "username": r[1], "role": r[2]} for r in result]
            return {"status": "success", "data": users}
        return {"status": "error", "message": "User not found"}
    except Exception as e:
        # Verbose error revealing DB details
        return Response(content=f"Database Error: {str(e)}\nQuery: {query}", status_code=500)

@app.get("/.env")
async def exposed_env():
    """Information Disclosure: Exposed .env file."""
    with open(".env", "r") as f:
        return Response(content=f.read(), media_type="text/plain")

@app.get("/admin", response_class=HTMLResponse)
async def admin_panel():
    """Information Disclosure: Exposed admin panel without auth."""
    return "<html><body><h1>Admin Control Panel</h1><p>Secret dashboard...</p></body></html>"

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return """
    <html><body>
        <h1>Login</h1>
        <form action="/api/login" method="POST">
            Username: <input type="text" name="username"><br>
            Password: <input type="password" name="password"><br>
            <input type="submit" value="Login">
        </form>
    </body></html>
    """

@app.post("/api/login")
async def login_api(username: str = Form(...), password: str = Form(...)):
    """Simple broken auth."""
    if username == "admin" and password == "' OR '1'='1":
        return {"status": "success", "token": "admin-token-123"}
    return {"status": "error", "message": "Invalid credentials"}

if __name__ == "__main__":
    print("Starting Vulnerable Target on http://localhost:5000")
    uvicorn.run(app, host="127.0.0.1", port=5000)
