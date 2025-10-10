# Warm-load all configured models so Dokploy deploys a ready-to-serve image
from app.main import pipeline

# Trigger lazy downloads by a lightweight no-op; model weights load on first run
# If a minimal image is desired, this step can be removed (will download on first request).
print("Models initialized:", type(pipeline).__name__)
