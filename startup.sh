# startup.sh
# Install necessary system libraries for WeasyPrint
apt-get update && apt-get install -y \
    libpango-1.0-0 \
    libcairo2 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    libgobject-2.0-0

# Start the Gunicorn server
gunicorn --bind=0.0.0.0:8000 wsgi:server
