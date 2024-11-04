# startup.sh
# Install necessary system libraries for WeasyPrint
apt-get update
apt-get install libgobject-2.0-0 libgdk-pixbuf2.0-0 libcairo2 libpango-1.0-0

# Start the Gunicorn server
gunicorn --bind=0.0.0.0:8000 wsgi:server
