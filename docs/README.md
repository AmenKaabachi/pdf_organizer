# 📚 Documentation

Welcome to the PDF Organizer AI documentation! This folder contains comprehensive guides for using, deploying, and understanding the project.

## 📖 Available Guides

### 🚀 [Deployment Guide](DEPLOYMENT_GUIDE.md)

Complete guide for deploying the PDF Organizer AI to various platforms:

- **Streamlit Cloud** (FREE, easiest)
- **Hugging Face Spaces** (FREE with GPU)
- **AWS EC2 / Azure VM** (full control)
- **Docker containers** (recommended for production)
- Model size comparisons and cost estimates
- Performance optimization tips

**Start here if you want to**: Deploy the app to a server or cloud platform

---

### 🛑 [Cancel Feature Documentation](CANCEL_FEATURE.md)

Technical documentation for the cancel/stop processing feature:

- How the cancel button works
- State management flow
- Use cases and scenarios
- Implementation details
- Testing checklist

**Start here if you want to**: Understand or modify the cancel functionality

---

### 🌍 [Multilingual Guide](MULTILINGUAL_GUIDE.md)

Comprehensive guide for multilingual document processing:

- Supported languages (100+)
- Language-specific model recommendations
- Real-world examples (Arabic+English, European languages, Asian languages)
- Performance benchmarks
- Best practices for mixed-language documents

**Start here if you want to**: Process documents in multiple languages

---

### 🏆 [Top 3 Models Comparison](TOP_3_MODELS_COMPARISON.md)

Technical deep dive on the best embedding models:

- **BAAI/bge-m3** (State-of-the-art, 66.1 MTEB)
- **intfloat/multilingual-e5-large** (Highest accuracy, 64.5 MTEB)
- **paraphrase-multilingual-mpnet-base-v2** (Best balance, 50.8 MTEB)
- Performance benchmarks and comparisons
- Decision matrices for model selection
- Real-world scenarios and expected results

**Start here if you want to**: Choose the right model for your use case

---

### ⚙️ [Model Setup Guide](MODEL_SETUP.md)

Instructions for downloading and configuring embedding models:

- One-time model download process
- Model caching and storage locations
- Troubleshooting download issues
- Switching between models

**Start here if you want to**: Set up models for the first time

---

## 🗂️ Quick Navigation

### By Use Case

**I want to...**

| Goal                             | Read This                                             |
| -------------------------------- | ----------------------------------------------------- |
| Deploy the app to production     | [Deployment Guide](DEPLOYMENT_GUIDE.md)               |
| Use with non-English documents   | [Multilingual Guide](MULTILINGUAL_GUIDE.md)           |
| Choose the best model            | [Top 3 Models Comparison](TOP_3_MODELS_COMPARISON.md) |
| Download models before first run | [Model Setup Guide](MODEL_SETUP.md)                   |
| Understand the cancel button     | [Cancel Feature](CANCEL_FEATURE.md)                   |

---

## 📊 Document Structure

```
docs/
├── README.md                          # This file - Documentation index
├── DEPLOYMENT_GUIDE.md                # Deployment instructions
├── MULTILINGUAL_GUIDE.md              # Multilingual processing guide
├── TOP_3_MODELS_COMPARISON.md         # Model comparison and benchmarks
├── MODEL_SETUP.md                     # Model download and setup
└── CANCEL_FEATURE.md                  # Cancel button technical docs
```

---

## 🆘 Getting Help

**Common Questions:**

1. **"Which model should I use?"**
   → See [Top 3 Models Comparison](TOP_3_MODELS_COMPARISON.md)

2. **"How do I deploy this?"**
   → See [Deployment Guide](DEPLOYMENT_GUIDE.md)

3. **"Can it handle Arabic/Chinese documents?"**
   → Yes! See [Multilingual Guide](MULTILINGUAL_GUIDE.md)

4. **"Model download is too slow/large"**
   → See [Model Setup Guide](MODEL_SETUP.md) for smaller alternatives

5. **"How does the cancel button work?"**
   → See [Cancel Feature Documentation](CANCEL_FEATURE.md)

---

## 🔗 Related Resources

- **Main README**: `../README.md` (project overview and quick start)
- **Source Code**: `../src/` (Python implementation)
- **Requirements**: `../requirements.txt` (Python dependencies)
- **Setup Script**: `../setup_models.py` (automated model download)

---

## 📝 Contributing

Found an issue or want to improve the documentation?

1. Edit the relevant `.md` file
2. Test your changes (ensure formatting is correct)
3. Submit a pull request or open an issue

---

## 📌 Last Updated

October 7, 2025

---

**Happy documenting! 📚✨**
