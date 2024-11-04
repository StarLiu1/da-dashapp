# startup.sh
# Install necessary system libraries for WeasyPrint
apt install weasyprint

# Start the Gunicorn server
gunicorn --bind=0.0.0.0:8000 wsgi:server
