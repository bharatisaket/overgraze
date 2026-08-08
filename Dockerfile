# Overgraze MCP server.
#
# One instance only. The tick barrier that makes agents act simultaneously lives
# in process memory, so a second replica would resolve its own ticks against the
# same database and agents would silently desynchronise. Scaling out means
# moving the barrier into the database first.
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY world.py harness.py store.py server.py deploy.py theory.py evolve.py ./

# State belongs on a mounted disk, not in the image layer.
ENV OVERGRAZE_DB=/data/overgraze.db
VOLUME ["/data"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s \
  CMD python -c "import urllib.request,os,sys; \
url='http://127.0.0.1:'+os.environ.get('PORT','8000')+'/healthz'; \
sys.exit(0 if urllib.request.urlopen(url, timeout=4).status==200 else 1)"

CMD ["python", "server.py"]
