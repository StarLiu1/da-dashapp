# wsgi.py
from app_main import server  # Note that this imports `server` from app.py

if __name__ == "__main__":
    server.run()

