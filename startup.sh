# startup.sh
# Install necessary system libraries for WeasyPrint
apt install python3-pip libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0 libjpeg-dev libopenjp2-7-dev libffi-dev gobject-2.0-0

# Start the Gunicorn server
gunicorn --bind=0.0.0.0:8000 wsgi:server
