# 🤖 AI Model Setup Guide

## ❓ Why Do I Need This?

The PDF Organizer uses a **438MB AI model** (`all-mpnet-base-v2`) to understand document content. This model is **NOT included in the repository** because:

- ❌ GitHub has a 100MB file size limit
- ❌ Including it would make the repository massive (438MB!)
- ❌ Every `git clone` would take forever
- ✅ Models are standardized and available from HuggingFace

## 🎯 Two Ways to Get the Model

### **Option 1: Pre-Download (RECOMMENDED)** ⚡

Run the setup script **once** before first use:

```bash
# Activate your virtual environment first
python setup_models.py
```

**What happens:**

- Downloads `all-mpnet-base-v2` (~438MB)
- Caches it to `~/.cache/huggingface/hub/`
- Takes ~5-10 minutes depending on internet speed
- ✅ **Only needs to be done ONCE**

**Advantages:**

- ✅ No delays when running the app
- ✅ Clear progress bars showing download status
- ✅ Verifies model works correctly
- ✅ Friendly error messages if something goes wrong

### **Option 2: Automatic Download** 🔄

Just run the app without the setup script:

```bash
streamlit run app.py
```

**What happens:**

- First run will download the model automatically
- You'll see download progress in the terminal
- Takes ~5-10 minutes on first run
- ⚠️ **App won't be usable until download completes**

**Disadvantages:**

- ⚠️ Unexpected wait time during first use
- ⚠️ Less user-friendly error handling
- ⚠️ May confuse users who don't expect the delay

## 📂 Where is the Model Stored?

The model is cached in your user directory:

**Windows:**

```
C:\Users\YourName\.cache\huggingface\hub\models--sentence-transformers--all-mpnet-base-v2\
```

**macOS/Linux:**

```
~/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/
```

**Git Ignore:** The `.gitignore` file prevents this cache from being committed to Git.

## 🔄 Model Updates

The model is versioned and cached. If you want to update to a newer version:

```bash
# Clear the cache
rm -rf ~/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2

# Re-run the setup script
python setup_models.py
```

## 🚨 Troubleshooting

### **Problem: Download Fails or Times Out**

```bash
# Try with a longer timeout
export HF_HUB_DOWNLOAD_TIMEOUT=300
python setup_models.py
```

### **Problem: No Internet Connection**

You can manually download the model on another machine and copy it:

1. Download from: https://huggingface.co/sentence-transformers/all-mpnet-base-v2
2. Copy to: `~/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/`
3. Verify with: `python setup_models.py`

### **Problem: Disk Space Issues**

The model requires ~500MB of free space. Check available space:

```bash
# Windows
dir "C:\Users\YourName\.cache\huggingface"

# macOS/Linux
du -sh ~/.cache/huggingface/
```

### **Problem: Permission Denied**

Run with administrator/sudo privileges:

```bash
# Windows (PowerShell as Administrator)
python setup_models.py

# macOS/Linux
sudo python setup_models.py
```

## 💡 For Developers: Alternative Models

If you want to use a different embedding model, edit `src/embeddings/embedding_generator.py`:

```python
# Available models
AVAILABLE_MODELS = {
    'all-MiniLM-L6-v2': {
        'size': 384,
        'download_size': '80MB',
        'speed': 'fast',
        'quality': 'good'
    },
    'all-mpnet-base-v2': {
        'size': 768,
        'download_size': '438MB',  # ← Currently selected
        'speed': 'medium',
        'quality': 'excellent'
    },
    'all-distilroberta-v1': {
        'size': 768,
        'download_size': '290MB',
        'speed': 'medium',
        'quality': 'very good'
    }
}
```

Then update `setup_models.py` to download your chosen model.

## 📊 Model Comparison

| Model                   | Size | Download | Speed  | Quality   | Best For              |
| ----------------------- | ---- | -------- | ------ | --------- | --------------------- |
| all-MiniLM-L6-v2        | 384D | 80MB     | Fast   | Good      | Large datasets        |
| **all-mpnet-base-v2**   | 768D | 438MB    | Medium | Excellent | **Quality priority**  |
| all-distilroberta-v1    | 768D | 290MB    | Medium | Very Good | Balanced performance  |
| paraphrase-MiniLM-L6-v2 | 384D | 80MB     | Fast   | Good      | Similar document sets |

**Current Selection:** `all-mpnet-base-v2` for best clustering quality!

## ✅ Verification

After running `setup_models.py`, verify the model is ready:

```bash
# Check if model exists
python -c "from sentence_transformers import SentenceTransformer; print('✓ Model ready!')"
```

Expected output:

```
✓ Model ready!
```

## 🎓 Why Not Include the Model?

**Technical Reasons:**

- Models are **binary files** (not suitable for version control)
- Multiple versions exist (would need to track all)
- Models are updated independently by HuggingFace
- Standard practice in ML projects

**Best Practices:**

- ✅ Store model references (e.g., `all-mpnet-base-v2`)
- ✅ Download from official sources
- ✅ Cache locally for fast access
- ✅ Document setup process clearly
- ❌ Don't commit large binary files to Git

## 📚 Additional Resources

- **HuggingFace Model Hub:** https://huggingface.co/sentence-transformers
- **Sentence Transformers Docs:** https://www.sbert.net/
- **Model Card:** https://huggingface.co/sentence-transformers/all-mpnet-base-v2

---

**Need Help?** Open an issue on GitHub or check the troubleshooting section above! 🚀
