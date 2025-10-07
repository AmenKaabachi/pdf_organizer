# 🚀 Deployment Guide - Model Handling

## Overview

This app uses **server-side AI models**. Users access a web interface; all processing happens on your server.

---

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (FREE, Easiest)

**Setup:**

```bash
# 1. Push to GitHub
git add .
git commit -m "Ready for deployment"
git push

# 2. Go to share.streamlit.io
# 3. Connect GitHub repo
# 4. Click Deploy
```

**Model Handling:**

- Model downloads automatically on first deploy (~5-10 minutes)
- Stored on Streamlit's servers
- All users share the same model instance
- Free tier: 1GB RAM (use smaller models)
- Paid tier: Up to 8GB RAM (can use BAAI/bge-m3)

**Recommended Model for Free Tier:**

```python
# Change in streamlit_app.py line 116:
default_model = 'all-MiniLM-L6-v2'  # 80MB, fits in free tier
# or
default_model = 'paraphrase-multilingual-MiniLM-L12-v2'  # 420MB
```

**Cost:** FREE (with limitations)

---

### Option 2: AWS EC2 / Azure VM (Full Control)

**Setup:**

```bash
# 1. Launch server (Ubuntu 22.04)
# Recommended: t3.medium (2 vCPU, 4GB RAM) or higher

# 2. Install dependencies
sudo apt update
sudo apt install python3-pip
pip install -r requirements.txt

# 3. Download models (one-time)
python setup_models.py

# 4. Run app
streamlit run app.py --server.port 8501
```

**Model Handling:**

- Models stored in `/home/ubuntu/.cache/huggingface/`
- Download once during setup (~10 minutes for 2.27GB)
- Persists across restarts
- All users share server's model

**Storage Requirements:**

- Small model: 500MB disk + 2GB RAM
- Large model: 3GB disk + 6GB RAM

**Cost:**

- AWS t3.medium: ~$30/month
- Can use spot instances: ~$8/month

---

### Option 3: Docker Container (Recommended for Production)

**Dockerfile:**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download model (baked into image)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Copy app
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

**Build & Deploy:**

```bash
# Build image (model downloads during build)
docker build -t pdf-organizer .

# Run container
docker run -p 8501:8501 pdf-organizer

# Deploy to cloud
docker push yourregistry/pdf-organizer
```

**Model Handling:**

- Model baked into Docker image (download once during build)
- Image size: ~3GB (includes model)
- Fast startup (no download needed)
- Can deploy to: AWS ECS, Azure Container Apps, Google Cloud Run

**Cost:**

- Google Cloud Run: Pay-per-use (~$5-20/month)
- AWS Fargate: ~$25-40/month

---

### Option 4: Hugging Face Spaces (FREE with GPU!)

**Setup:**

```bash
# 1. Create space on huggingface.co/spaces
# 2. Push code to space repo
# 3. Add app.py as entry point
```

**Model Handling:**

- Models auto-download on space startup
- Cached on HF servers
- Free GPU available (faster inference)
- Unlimited users

**Cost:** FREE (community tier) or $9/month (Pro with better hardware)

---

## 📦 Model Size Comparison

| Model                                 | Download Size | RAM Usage | Quality   | Languages |
| ------------------------------------- | ------------- | --------- | --------- | --------- |
| all-MiniLM-L6-v2                      | 80MB          | 1GB       | Good      | English   |
| paraphrase-multilingual-MiniLM-L12-v2 | 420MB         | 2GB       | Very Good | 50+       |
| paraphrase-multilingual-mpnet-base-v2 | 1.0GB         | 4GB       | Excellent | 50+       |
| BAAI/bge-m3                           | 2.27GB        | 6GB       | Best      | 100+      |

---

## 🔧 Changing Default Model for Deployment

**For Free Tier / Limited Resources:**

Edit `src/app/streamlit_app.py` line 116:

```python
# Change from:
default_model = 'BAAI/bge-m3'  # 2.27GB

# To (for free tier):
default_model = 'all-MiniLM-L6-v2'  # 80MB, English only

# Or (for multilingual, medium size):
default_model = 'paraphrase-multilingual-MiniLM-L12-v2'  # 420MB, 50+ languages
```

---

## ⚡ Performance Considerations

### Server Capacity Planning:

**Light Usage (1-10 concurrent users):**

- 2GB RAM, 1 vCPU
- Model: all-MiniLM-L6-v2 (80MB)
- Cost: ~$10-15/month

**Medium Usage (10-50 concurrent users):**

- 4GB RAM, 2 vCPU
- Model: paraphrase-multilingual-MiniLM-L12-v2 (420MB)
- Cost: ~$30-40/month

**Heavy Usage (50+ concurrent users):**

- 8GB RAM, 4 vCPU
- Model: BAAI/bge-m3 (2.27GB)
- Load balancer + multiple instances
- Cost: ~$100+/month

---

## 🔒 Security Considerations

**When Deployed:**

1. Users' PDFs are uploaded to your server
2. Processed in memory (not saved to disk by default)
3. Results sent back to user
4. No data sent to external APIs (fully self-contained)

**To enhance security:**

- Enable HTTPS (SSL certificate)
- Add authentication (Streamlit supports this)
- Implement file size limits
- Add virus scanning for uploaded files

---

## 📊 Cost Comparison Summary

| Platform               | Model Size | Monthly Cost | Setup Time |
| ---------------------- | ---------- | ------------ | ---------- |
| Streamlit Cloud (Free) | <500MB     | $0           | 5 min      |
| Streamlit Cloud (Paid) | Any        | $20          | 5 min      |
| Hugging Face Spaces    | Any        | $0-9         | 10 min     |
| AWS EC2 (t3.medium)    | Any        | $30          | 30 min     |
| Google Cloud Run       | Any        | $5-20\*      | 20 min     |
| Docker on VPS          | Any        | $10-50       | 45 min     |

\*Pay-per-use, actual cost depends on traffic

---

## ✅ Recommended Setup

**For Testing/Personal Use:**

- Platform: Streamlit Cloud (free tier)
- Model: `paraphrase-multilingual-MiniLM-L12-v2` (420MB)
- Cost: $0
- Setup time: 5 minutes

**For Production/Business:**

- Platform: AWS EC2 or Google Cloud Run
- Model: `BAAI/bge-m3` (2.27GB)
- Cost: $30-40/month
- Setup time: 30 minutes

---

## 🚨 Important Notes

1. **Model downloads happen server-side** - Users never download models
2. **Download happens once** - Model cached on server permanently
3. **All users share one model** - No per-user downloads
4. **Users only access web interface** - No installation required
5. **Processing is server-side** - Works on any device (phone, tablet, etc.)

---

## 🆘 Need Help?

**Model too large for your platform?**
→ Change default model to smaller one (see above)

**App crashing due to memory?**
→ Upgrade server RAM or use smaller model

**Want to use external API instead?**
→ Contact for API integration guide (OpenAI, Cohere, etc.)
