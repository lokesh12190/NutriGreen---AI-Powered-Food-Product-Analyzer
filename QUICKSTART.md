# 🚀 NutriGreen Quick Start Guide

Get NutriGreen up and running in under 10 minutes!

---

## ⚡ Prerequisites Checklist

Before you begin, ensure you have:

- [ ] Python 3.8 or higher installed
- [ ] Git installed
- [ ] (Optional) CUDA-capable GPU with 6GB+ VRAM
- [ ] 20GB free disk space
- [ ] Stable internet connection for model downloads

---

## 📦 Installation (5 Steps)

### Step 1: Clone and Navigate
```bash
git clone https://github.com/yourusername/nutrigreen.git
cd nutrigreen
```

### Step 2: Create Virtual Environment
```bash
# Option A: Using conda (recommended)
conda create -n nutrigreen python=3.9 -y
conda activate nutrigreen

# Option B: Using venv
python -m venv nutrigreen_env
source nutrigreen_env/bin/activate  # Linux/Mac
# nutrigreen_env\Scripts\activate   # Windows
```

### Step 3: Install PyTorch
```bash
# For CUDA 11.8 (check your CUDA version first)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only (slower, but works without GPU)
pip install torch torchvision torchaudio
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Clone YOLOv5
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
cd ..
```

---

## 🔧 Configuration (2 Minutes)

### 1. Create Environment File

Create a `.env` file in the project root:

```bash
touch .env  # Linux/Mac
# type nul > .env  # Windows
```

Add your OpenAI API key (optional, for Premium mode):

```env
OPENAI_API_KEY=sk-your-key-here
```

### 2. Update File Paths

Open `nutrigreen_app.py` and update lines 62-66 with your paths:

```python
YOLO_WEIGHTS = "./models/best.pt"              # Your YOLOv5 weights
YOLO_REPO = "./yolov5"                         # YOLOv5 directory
DATABASE_PATH = "./db/nutrigreen_products.db"  # Database file
EMBEDDINGS_INDEX = "./db/product_embeddings.faiss"
EMBEDDINGS_METADATA = "./db/product_metadata.pkl"
```

---

## 📥 Download Required Files

You'll need these files (contact maintainer or check releases):

1. **YOLOv5 Weights** (`best.pt`) → Place in `./models/`
2. **Database** (`nutrigreen_products.db`) → Place in `./db/`
3. **Embeddings** (`product_embeddings.faiss`) → Place in `./db/`
4. **Metadata** (`product_metadata.pkl`) → Place in `./db/`

Create directories if they don't exist:
```bash
mkdir -p models db data
```

---

## 🎯 Launch Application

```bash
streamlit run nutrigreen_app.py
```

The app will open automatically in your browser at `http://localhost:8501`

**First Launch Tips:**
- Models will download on first use (~5-10 minutes)
- Moondream2: ~4GB
- LLaVA-1.5: ~15GB (downloads in 4-bit quantized form)
- Be patient during initial model loading

---

## 🧪 Quick Test

### Test 1: Check Installation
1. Open browser to `http://localhost:8501`
2. Navigate to "Home" page
3. Check Database Stats in sidebar (should show 7,271 products)

### Test 2: Analyze an Image
1. Go to "Analyze Image" page
2. Upload any food product image
3. Select "Quick" mode
4. Click "Analyze"
5. Wait for results (~3-5 seconds)

### Test 3: Database Search
1. Go to "Database Explorer"
2. Search for "milk"
3. Should return multiple results
4. Click on any product to view details

---

## 🎨 First-Time User Guide

### Understanding the Modes

**🚀 Quick Mode (Moondream2)**
- **When to use**: Fast categorization, basic analysis
- **Speed**: 2-3 seconds
- **VRAM**: ~2GB
- **Best for**: Quick scans, checking categories

**⚡ Standard Mode (LLaVA-1.5)**
- **When to use**: Detailed analysis, ingredient lists
- **Speed**: 5-8 seconds
- **VRAM**: ~4GB
- **Best for**: Comprehensive product analysis

**💎 Premium Mode (GPT-4o)**
- **When to use**: Professional use, highest accuracy
- **Speed**: 3-5 seconds
- **Cost**: Uses OpenAI API (pay per use)
- **Best for**: Critical decisions, detailed nutrition info
- **Requires**: OpenAI API key in `.env` file

---

## 📱 Main Features Tour

### 1. Analyze Image
Upload a product photo and get instant analysis including:
- Product category detection
- Ingredient identification
- Nutritional insights
- Usage suggestions

### 2. Database Explorer
Search through 7,271+ products:
- Filter by category, organic, vegan
- View detailed nutrition facts
- Export results to CSV

### 3. Compare Products
Side-by-side comparison of 2-4 products:
- Nutritional breakdown
- Interactive charts
- Best choice recommendations

### 4. Nutrition Calculator
Plan your meals:
- Add products to daily plan
- Track macros (protein, carbs, fat)
- Visualize daily intake
- Export meal plans

### 5. Allergen Alerts
Safety first:
- Set your allergens
- Filter safe products
- Get instant warnings

### 6. Recommendations
Find similar products:
- Semantic search
- "Show me alternatives"
- Filter by preferences

---

## 🐛 Common Issues & Solutions

### Issue: "CUDA out of memory"
**Solution**: 
```python
# In vision_manager.py, the system automatically manages VRAM
# If still having issues, use Quick mode (Moondream2)
# Or run on CPU by removing CUDA in PyTorch install
```

### Issue: "Module not found"
**Solution**:
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Check if virtual environment is activated
conda activate nutrigreen  # or source nutrigreen_env/bin/activate
```

### Issue: "Database file not found"
**Solution**:
```bash
# Ensure database files are in correct location
ls ./db/
# Should show: nutrigreen_products.db, product_embeddings.faiss, product_metadata.pkl
```

### Issue: "Premium mode not working"
**Solution**:
```bash
# Check .env file has correct API key
cat .env
# Should show: OPENAI_API_KEY=sk-...

# Test API key
python -c "import openai; openai.api_key='YOUR_KEY'; print('OK')"
```

### Issue: Models taking too long to load
**Solution**:
- First load always takes longer (downloading models)
- Subsequent loads use cache
- Quick mode: ~30 seconds first load
- Standard mode: ~5 minutes first load
- Check internet connection

---

## 💡 Tips for Best Results

### Image Quality
- ✅ Good lighting
- ✅ Clear focus on product
- ✅ Product label visible
- ✅ Resolution: 640x640 or higher
- ❌ Avoid: blurry, too dark, obstructed

### Mode Selection
- **Quick scans**: Use Quick mode
- **Detailed analysis**: Use Standard mode
- **Professional/critical**: Use Premium mode (with API key)

### Performance Optimization
- Close other GPU-heavy applications
- Use Quick mode for batch processing
- Standard/Premium for important analyses
- Monitor VRAM usage in Streamlit sidebar

---

## 📚 Next Steps

Now that you're set up, explore:

1. **📓 Jupyter Notebooks**: Check out notebooks for data exploration
   ```bash
   jupyter notebook notebooks/
   ```

2. **📖 Full Documentation**: Read the complete [README.md](README.md)

3. **🎓 Tutorial Videos**: Watch demo videos (if available)

4. **🤝 Community**: Join discussions on GitHub

---

## 🆘 Getting Help

If you encounter issues:

1. **Check Documentation**: [README.md](README.md) has detailed info
2. **GitHub Issues**: Search existing issues or create new one
3. **Discussions**: Ask questions in GitHub Discussions
4. **Email**: Contact maintainer at your.email@example.com

---

## ✅ Success Checklist

Before reporting issues, verify:

- [ ] Python 3.8+ installed and working
- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip list`)
- [ ] YOLOv5 cloned and setup
- [ ] Database files in correct location
- [ ] File paths updated in `nutrigreen_app.py`
- [ ] Models can access internet for first download
- [ ] GPU drivers up to date (if using GPU)
- [ ] `.env` file created (if using Premium mode)

---

## 🎉 You're Ready!

Congratulations! You're all set to analyze food products with AI.

**Recommended First Project:**
1. Upload 5 different product images
2. Try all three analysis modes
3. Compare 2-3 similar products
4. Create a meal plan with Nutrition Calculator
5. Set up allergen alerts for your dietary needs

**Have fun exploring! 🥗🚀**

---

<div align="center">

**Questions? Issues? Feedback?**

[Open an Issue](https://github.com/yourusername/nutrigreen/issues) • [Start a Discussion](https://github.com/yourusername/nutrigreen/discussions)

Made with ❤️ for healthier food choices

</div>
