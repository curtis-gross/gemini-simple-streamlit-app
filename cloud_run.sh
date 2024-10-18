gcloud run deploy genai-spawn --source . --platform managed --region us-central1 \
  --max-instances 1 --memory 2Gi --cpu 2 --allow-unauthenticated --session-affinity \
  --port 8501 --timeout=3600
