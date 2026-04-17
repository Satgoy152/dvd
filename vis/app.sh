export HF_HOME="/nfs/turbo/coe-jjparkcv-medium/satyam/.cache/huggingface"

streamlit run vis/app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true --server.enableCORS false --server.enableXsrfProtection false