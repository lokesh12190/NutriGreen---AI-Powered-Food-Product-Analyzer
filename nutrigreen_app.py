"""
NutriGreen - Complete Food Product Analyzer
Multi-page Streamlit app with vision AI, database, and advanced features
"""

import streamlit as st
from streamlit_option_menu import option_menu
import torch
from PIL import Image
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import re

# Import our modules
from vision_manager import VisionManager
from nutrigreen_analysis_advanced import (
    ProductDatabase,
    ComparativeAnalyzer,
    NutritionCalculator,
    AllergenFilter,
    RecommendationEngine
)

# Page config
st.set_page_config(
    page_title="NutriGreen - Food Analyzer",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert {
        border-radius: 10px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

YOLO_WEIGHTS = r"C:\Users\lokes\Desktop\ironhack\final_project\dataset\dataset\runs\nutrigreen_exp\weights\best.pt"
YOLO_REPO = r"C:\Users\lokes\Desktop\ironhack\final_project\dataset\dataset\yolov5"
DATABASE_PATH = r"C:\Users\lokes\Desktop\ironhack\final_project\final_project version2\db\nutrigreen_products.db"
EMBEDDINGS_INDEX = r"C:\Users\lokes\Desktop\ironhack\final_project\final_project version2\db\product_embeddings.faiss"
EMBEDDINGS_METADATA = r"C:\Users\lokes\Desktop\ironhack\final_project\final_project version2\db\product_metadata.pkl"

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

@st.cache_resource
def load_yolo_model():
    """Load YOLO model (cached)"""
    sys.path.insert(0, str(YOLO_REPO))
    try:
        model = torch.hub.load(
            str(YOLO_REPO),
            'custom',
            path=str(YOLO_WEIGHTS),
            source='local',
            force_reload=False
        )
        model.conf = 0.25
        return model
    except Exception as e:
        st.error(f"Error loading YOLO: {e}")
        return None

@st.cache_resource
def load_vision_manager():
    """Load Vision Manager (cached)"""
    try:
        return VisionManager()
    except Exception as e:
        st.error(f"Error loading Vision Manager: {e}")
        return None

@st.cache_resource
def load_database_tools():
    """Load database and analysis tools (cached)"""
    db = ProductDatabase(DATABASE_PATH)
    comparator = ComparativeAnalyzer(db)
    nutrition_calc = NutritionCalculator(db)
    allergen_filter = AllergenFilter(db)
    recommender = RecommendationEngine(db, EMBEDDINGS_INDEX, EMBEDDINGS_METADATA)
    
    return db, comparator, nutrition_calc, allergen_filter, recommender

# Load everything
yolo_model = load_yolo_model()
vision_manager = load_vision_manager()
db, comparator, nutrition_calc, allergen_filter, recommender = load_database_tools()

# Update OpenAI API key if user provided one
def update_api_key():
    if 'openai_api_key' in st.session_state and vision_manager:
        import os
        os.environ['OPENAI_API_KEY'] = st.session_state['openai_api_key']

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #2E7D32;'>🥗 NutriGreen</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # API Key input for Premium mode
    with st.expander("🔑 Premium Mode Settings"):
        user_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to enable Premium mode (GPT-4o Vision)",
            placeholder="sk-..."
        )
        if user_api_key:
            st.success("✅ API key configured!")
            # Store in session state
            st.session_state['openai_api_key'] = user_api_key
        else:
            st.info("💡 Premium mode requires an OpenAI API key")
    
    st.markdown("---")
    
    selected = option_menu(
        menu_title="Navigation",
        options=[
            "Home",
            "Analyze Image",
            "Database Explorer",
            "Compare Products",
            "Nutrition Calculator",
            "Allergen Alerts",
            "Recommendations"
        ],
        icons=[
            "house",
            "camera",
            "database",
            "bar-chart",
            "calculator",
            "exclamation-triangle",
            "star"
        ],
        menu_icon="list",
        default_index=0,
    )
    
    st.markdown("---")
    st.markdown("### 📊 Database Stats")
    
    # Get database stats
    try:
        stats = db.search_products(limit=1000000)  # Get all
        st.metric("Total Products", len(stats))
        
        organic_count = sum(1 for p in stats if p.get('is_organic'))
        st.metric("Organic Products", organic_count)
        
        vegan_count = sum(1 for p in stats if p.get('is_vegan_vegetarian'))
        st.metric("Vegan Products", vegan_count)
    except:
        st.info("Database stats unavailable")

# ============================================================================
# PAGE: HOME
# ============================================================================

if selected == "Home":
    st.markdown("<h1 class='main-header'>🥗 NutriGreen Food Analyzer</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to NutriGreen!
    
    Your intelligent food product analyzer powered by AI and computer vision.
    
    **Features:**
    - 📸 **Image Analysis** - Upload product images for instant analysis
    - 🔍 **Database Explorer** - Browse 10,000+ products
    - ⚖️ **Product Comparison** - Compare multiple products side-by-side
    - 🧮 **Nutrition Calculator** - Calculate daily intake percentages
    - 🚨 **Allergen Alerts** - Filter products by allergens
    - ⭐ **Recommendations** - Find healthier alternatives
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**🎯 Quick Mode**\n\nMoondream2 - 2-3s\n\nOffline, Fast")
    
    with col2:
        st.info("**⚡ Standard Mode**\n\nLLaVA-1.5 - 5-7s\n\nOffline, Accurate")
    
    with col3:
        st.info("**✨ Premium Mode**\n\nGPT-4o Vision\n\nBest Quality")
    
    st.markdown("---")
    st.success("👈 Choose a feature from the sidebar to get started!")

# ============================================================================
# PAGE: ANALYZE IMAGE
# ============================================================================

elif selected == "Analyze Image":
    st.markdown("<h1 class='main-header'>📸 Analyze Product Image</h1>", unsafe_allow_html=True)
    
    # Vision mode selection
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        vision_mode = st.selectbox(
            "Select Analysis Mode",
            ["quick", "standard", "premium"],
            format_func=lambda x: {
                "quick": "🎯 Quick (Moondream2)",
                "standard": "⚡ Standard (LLaVA)",
                "premium": "✨ Premium (GPT-4o)"
            }[x]
        )
    
    with col2:
        mode_info = {
            "quick": "Fast offline (~2-3s)",
            "standard": "Better quality (~5-7s)",
            "premium": "Best quality (requires API key)"
        }
        st.info(mode_info[vision_mode])
    
    with col3:
        save_to_db = st.checkbox("Save to Database", value=False)
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Upload Product Image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of the product showing labels"
    )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", width="stretch")
            
            # Analyze button
            if st.button("🔍 Analyze Product", type="primary", width="stretch"):
                with st.spinner(f"Analyzing with {vision_mode} mode..."):
                    # YOLO Detection
                    yolo_results = yolo_model(np.array(image))
                    detections = []
                    
                    for pred in yolo_results.pred[0]:
                        x1, y1, x2, y2, conf, cls = pred.tolist()
                        label_names = {
                            0: 'NutriScore_A', 1: 'NutriScore_B', 2: 'NutriScore_C',
                            3: 'NutriScore_D', 4: 'NutriScore_E', 5: 'BIO', 6: 'V-Label'
                        }
                        detections.append({
                            'label': label_names[int(cls)],
                            'confidence': float(conf),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
                    
                    # Vision Analysis
                    result = vision_manager.analyze_image(
                        image,
                        mode=vision_mode,
                        yolo_detections=detections
                    )
                    
                    # Store results in session state
                    st.session_state['last_analysis'] = {
                        'detections': detections,
                        'result': result,
                        'image': image,
                        'filename': uploaded_file.name
                    }
        
        # Show results if available
        if 'last_analysis' in st.session_state:
            analysis = st.session_state['last_analysis']
            detections = analysis['detections']
            result = analysis['result']
            
            with col2:
                st.markdown("### 🎯 Analysis Results")
                
                # Calculate health score
                nutriscore = None
                is_organic = False
                is_vegan = False
                health_score = 50.0
                
                for det in detections:
                    if 'NutriScore' in det['label']:
                        nutriscore = det['label']
                        score_map = {
                            'NutriScore_A': 100, 'NutriScore_B': 80, 'NutriScore_C': 60,
                            'NutriScore_D': 40, 'NutriScore_E': 20
                        }
                        health_score = score_map.get(nutriscore, 50)
                    elif det['label'] == 'BIO':
                        is_organic = True
                    elif det['label'] == 'V-Label':
                        is_vegan = True
                
                # Show scores
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Health Score", f"{health_score:.0f}/100")
                with col_b:
                    if nutriscore:
                        st.metric("NutriScore", nutriscore.replace('NutriScore_', ''))
                    else:
                        st.metric("NutriScore", "N/A")
                with col_c:
                    badges = []
                    if is_organic:
                        badges.append("🌱")
                    if is_vegan:
                        badges.append("🌿")
                    st.metric("Labels", " ".join(badges) if badges else "None")
                
                st.markdown("---")
                
                # Show detections
                if detections:
                    st.markdown("**🏷️ Detected Labels:**")
                    for det in detections:
                        if 'NutriScore' in det['label']:
                            st.success(f"✅ {det['label']} (Confidence: {det['confidence']:.2%})")
                        elif det['label'] == 'BIO':
                            st.success(f"🌱 Organic Label (Confidence: {det['confidence']:.2%})")
                        elif det['label'] == 'V-Label':
                            st.success(f"🌿 Vegan Label (Confidence: {det['confidence']:.2%})")
                else:
                    st.info("No labels detected")
                
                st.markdown("---")
                
                # Show vision analysis
                if 'error' not in result:
                    st.markdown("**🤖 AI Analysis:**")
                    st.write(result.get('response', 'No response'))
                    
                    st.markdown("---")
                    st.caption(f"⏱️ Analysis time: {result.get('analysis_time', 0):.2f}s | Mode: {result.get('mode', 'unknown')}")
                else:
                    st.error(f"Analysis error: {result['error']}")

# ============================================================================
# PAGE: DATABASE EXPLORER
# ============================================================================

elif selected == "Database Explorer":
    st.markdown("<h1 class='main-header'>🗄️ Product Database</h1>", unsafe_allow_html=True)
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        filter_nutriscore = st.selectbox(
            "NutriScore",
            ["All", "NutriScore_A", "NutriScore_B", "NutriScore_C", "NutriScore_D", "NutriScore_E"]
        )
    
    with col2:
        filter_organic = st.checkbox("Organic Only")
    
    with col3:
        filter_vegan = st.checkbox("Vegan Only")
    
    with col4:
        min_health = st.slider("Min Health Score", 0, 100, 0)
    
    # Search
    brand_search = st.text_input("🔍 Search by Brand")
    
    # Get products
    products = db.search_products(
        brand=brand_search if brand_search else None,
        nutriscore=filter_nutriscore if filter_nutriscore != "All" else None,
        is_organic=filter_organic if filter_organic else None,
        is_vegan=filter_vegan if filter_vegan else None,
        min_health_score=min_health,
        limit=50
    )
    
    st.markdown(f"### Found {len(products)} products")
    
    if products:
        # Display as cards
        for i in range(0, len(products), 3):
            cols = st.columns(3)
            
            for j, col in enumerate(cols):
                if i + j < len(products):
                    product = products[i + j]
                    
                    with col:
                        with st.container():
                            st.markdown(f"**{product.get('product_name', 'Unknown')}**")
                            st.caption(f"Brand: {product.get('brand', 'Unknown')}")
                            
                            if product.get('nutriscore'):
                                st.success(f"NutriScore: {product['nutriscore']}")
                            
                            st.metric("Health Score", f"{product.get('health_score', 0):.0f}/100")
                            
                            badges = []
                            if product.get('is_organic'):
                                badges.append("🌱 Organic")
                            if product.get('is_vegan_vegetarian'):
                                badges.append("🌿 Vegan")
                            
                            if badges:
                                st.write(" | ".join(badges))
                            
                            st.markdown("---")
    else:
        st.info("No products found matching your criteria")

# ============================================================================
# PAGE: COMPARE PRODUCTS
# ============================================================================

elif selected == "Compare Products":
    st.markdown("<h1 class='main-header'>⚖️ Compare Products</h1>", unsafe_allow_html=True)
    
    st.info("📸 Upload 2-5 product images to compare them side-by-side")
    
    # Image upload
    uploaded_files = st.file_uploader(
        "Upload Product Images",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload 2-5 images to compare"
    )
    
    if uploaded_files:
        num_images = len(uploaded_files)
        
        if num_images < 2:
            st.warning("Please upload at least 2 images to compare")
        elif num_images > 5:
            st.warning("Maximum 5 images can be compared. Only the first 5 will be used.")
            uploaded_files = uploaded_files[:5]
        else:
            # Select vision mode for analysis
            vision_mode = st.selectbox(
                "Analysis Mode",
                ["quick", "standard", "premium"],
                format_func=lambda x: {
                    "quick": "🎯 Quick (Fast)",
                    "standard": "⚡ Standard (Accurate)",
                    "premium": "✨ Premium (Best - needs API key)"
                }[x],
                help="Quick mode is faster, Premium gives best results but needs OpenAI API key"
            )
            
            if st.button("🔍 Compare Products", type="primary", width="stretch"):
                with st.spinner(f"Analyzing {len(uploaded_files)} products..."):
                    
                    analyzed_products = []
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    
                    for idx, uploaded_file in enumerate(uploaded_files):
                        # Load image
                        image = Image.open(uploaded_file).convert('RGB')
                        
                        # YOLO Detection
                        yolo_results = yolo_model(np.array(image))
                        detections = []
                        
                        for pred in yolo_results.pred[0]:
                            x1, y1, x2, y2, conf, cls = pred.tolist()
                            label_names = {
                                0: 'NutriScore_A', 1: 'NutriScore_B', 2: 'NutriScore_C',
                                3: 'NutriScore_D', 4: 'NutriScore_E', 5: 'BIO', 6: 'V-Label'
                            }
                            detections.append({
                                'label': label_names[int(cls)],
                                'confidence': float(conf),
                                'bbox': [int(x1), int(y1), int(x2), int(y2)]
                            })
                        
                        # Vision Analysis
                        result = vision_manager.analyze_image(
                            image,
                            mode=vision_mode,
                            yolo_detections=detections
                        )
                        
                        # Calculate health score
                        nutriscore = None
                        is_organic = False
                        is_vegan = False
                        health_score = 50.0
                        
                        for det in detections:
                            if 'NutriScore' in det['label']:
                                nutriscore = det['label']
                                score_map = {
                                    'NutriScore_A': 100, 'NutriScore_B': 80, 'NutriScore_C': 60,
                                    'NutriScore_D': 40, 'NutriScore_E': 20
                                }
                                health_score = score_map.get(nutriscore, 50)
                            elif det['label'] == 'BIO':
                                is_organic = True
                            elif det['label'] == 'V-Label':
                                is_vegan = True
                        
                        analyzed_products.append({
                            'id': idx + 1,
                            'image': image,
                            'filename': uploaded_file.name,
                            'nutriscore': nutriscore,
                            'is_organic': is_organic,
                            'is_vegan': is_vegan,
                            'health_score': health_score,
                            'detections': detections,
                            'analysis': result.get('response', 'No analysis available'),
                            'analysis_time': result.get('analysis_time', 0)
                        })
                        
                        # Update progress
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    progress_bar.empty()
                
                # Display comparison
                st.markdown("---")
                st.markdown("## 📊 Comparison Results")
                
                # Show images in grid
                st.markdown("### Product Images")
                cols = st.columns(len(analyzed_products))
                for idx, (col, product) in enumerate(zip(cols, analyzed_products)):
                    with col:
                        st.image(product['image'], caption=f"Product {idx + 1}", width=200)
                
                st.markdown("---")
                
                # Comparison Table
                st.markdown("### 📋 Side-by-Side Comparison")
                
                comparison_data = []
                for product in analyzed_products:
                    comparison_data.append({
                        'Product': f"Product {product['id']}",
                        'Filename': product['filename'],
                        'NutriScore': product['nutriscore'] or 'N/A',
                        'Health Score': f"{product['health_score']:.0f}/100",
                        'Organic': '🌱' if product['is_organic'] else '❌',
                        'Vegan': '🌿' if product['is_vegan'] else '❌'
                    })
                
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, width="stretch", hide_index=True)
                
                # Winner
                winner = max(analyzed_products, key=lambda x: x['health_score'])
                st.success(f"🏆 **Winner:** Product {winner['id']} ({winner['filename']}) - Health Score: {winner['health_score']:.0f}/100")
                
                # Radar Chart
                st.markdown("### 📈 Visual Comparison")
                
                fig = go.Figure()
                
                categories = ['Health Score', 'Organic', 'Vegan', 'NutriScore']
                
                nutriscore_map = {
                    'NutriScore_A': 100, 'NutriScore_B': 80, 'NutriScore_C': 60,
                    'NutriScore_D': 40, 'NutriScore_E': 20, None: 50
                }
                
                for product in analyzed_products:
                    values = [
                        product['health_score'],
                        100 if product['is_organic'] else 0,
                        100 if product['is_vegan'] else 0,
                        nutriscore_map.get(product['nutriscore'], 50)
                    ]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        name=f"Product {product['id']}",
                        fill='toself'
                    ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    title="Product Comparison",
                    showlegend=True,
                    height=500
                )
                
                st.plotly_chart(fig, width="stretch")
                
                # Detailed Analysis
                st.markdown("---")
                st.markdown("### 🔍 Detailed Analysis")
                
                for product in analyzed_products:
                    with st.expander(f"Product {product['id']}: {product['filename']}"):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.image(product['image'], width=250)
                            
                            st.markdown("**Detected Labels:**")
                            if product['detections']:
                                for det in product['detections']:
                                    st.write(f"• {det['label']} ({det['confidence']:.1%})")
                            else:
                                st.write("No labels detected")
                        
                        with col2:
                            st.markdown("**AI Analysis:**")
                            st.write(product['analysis'])
                            
                            st.markdown("---")
                            st.caption(f"Analysis time: {product['analysis_time']:.2f}s")
                            
                            st.markdown("**Scores:**")
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Health Score", f"{product['health_score']:.0f}")
                            with col_b:
                                st.metric("Organic", "Yes" if product['is_organic'] else "No")
                            with col_c:
                                st.metric("Vegan", "Yes" if product['is_vegan'] else "No")

# ============================================================================
# PAGE: NUTRITION CALCULATOR
# ============================================================================

elif selected == "Nutrition Calculator":
    st.markdown("<h1 class='main-header'>🧮 Nutrition Calculator</h1>", unsafe_allow_html=True)
    
    st.info("📸 Upload a product image to analyze its nutrition facts")
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Upload Product Image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload image showing nutrition facts table",
        key="nutrition_upload"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Product Image", width="stretch")
            
            # Settings
            st.markdown("### ⚙️ Settings")
            serving_size = st.number_input("Serving Size (g)", min_value=1, value=100, step=10)
            standard = st.selectbox("Nutrition Standard", ["EU", "US"])
            
            if st.button("🧮 Calculate Nutrition", type="primary", width="stretch"):
                with st.spinner("Analyzing nutrition facts..."):
                    # Import OCR
                    try:
                        from paddleocr import PaddleOCR
                        import re
                        
                        # Initialize OCR
                        ocr = PaddleOCR(use_textline_orientation=True, lang='en')
                        
                        # Extract text
                        img_array = np.array(image)
                        result = ocr.ocr(img_array, cls=True)
                        
                        # Extract nutrition values
                        text_parts = []
                        if result and result[0]:
                            for line in result[0]:
                                text_parts.append(line[1][0])
                        
                        full_text = ' '.join(text_parts).lower()
                        
                        # Nutrition patterns
                        nutrition_patterns = {
                            'calories': r'(calories?|kcal|energie|energy)[:\\s]*([0-9]+[.,]?[0-9]*)',
                            'total_fat': r'(fat|fett|graisses?|matières?\\s*grasses?)[:\\s]*([0-9]+[.,]?[0-9]*)\\s*g',
                            'saturated_fat': r'(saturated|gesättigt|saturées?)[:\\s]*([0-9]+[.,]?[0-9]*)\\s*g',
                            'carbohydrates': r'(carbohydrat|glucid|kohlenhydrat)[:\\s]*([0-9]+[.,]?[0-9]*)\\s*g',
                            'sugars': r'(sugar|zucker|sucres?)[:\\s]*([0-9]+[.,]?[0-9]*)\\s*g',
                            'protein': r'(protein|eiweiß|protéin)[:\\s]*([0-9]+[.,]?[0-9]*)\\s*g',
                            'salt': r'(salt|salz|sel)[:\\s]*([0-9]+[.,]?[0-9]*)\\s*g',
                            'fiber': r'(fiber|fibre|ballaststoff)[:\\s]*([0-9]+[.,]?[0-9]*)\\s*g'
                        }
                        
                        nutrition_data = {}
                        for nutrient, pattern in nutrition_patterns.items():
                            match = re.search(pattern, full_text, re.IGNORECASE)
                            if match:
                                try:
                                    value = float(match.group(2).replace(',', '.'))
                                    nutrition_data[nutrient] = value
                                except:
                                    pass
                        
                        # Store in session state
                        st.session_state['nutrition_data'] = nutrition_data
                        st.session_state['serving_size'] = serving_size
                        st.session_state['standard'] = standard
                        
                    except Exception as e:
                        st.error(f"Error extracting nutrition facts: {e}")
                        st.info("💡 Make sure the image clearly shows the nutrition facts table")
        
        # Display results
        if 'nutrition_data' in st.session_state and st.session_state['nutrition_data']:
            nutrition_data = st.session_state['nutrition_data']
            serving_size = st.session_state['serving_size']
            standard = st.session_state['standard']
            
            with col2:
                st.markdown("### 📊 Nutrition Facts")
                st.markdown(f"**Per {serving_size}g serving**")
                
                # Daily values
                if standard == "EU":
                    daily_values = {
                        'calories': 2000, 'total_fat': 70, 'saturated_fat': 20,
                        'carbohydrates': 260, 'sugars': 90, 'protein': 50,
                        'salt': 6, 'fiber': 25
                    }
                else:  # US
                    daily_values = {
                        'calories': 2000, 'total_fat': 78, 'saturated_fat': 20,
                        'carbohydrates': 275, 'sugars': 50, 'protein': 50,
                        'salt': 5, 'fiber': 28
                    }
                
                # Calculate per serving
                multiplier = serving_size / 100
                
                # Create table
                nutrition_table = []
                for nutrient, value in nutrition_data.items():
                    amount = value * multiplier
                    daily_value = daily_values.get(nutrient, 100)
                    percentage = (amount / daily_value) * 100
                    
                    nutrition_table.append({
                        'Nutrient': nutrient.replace('_', ' ').title(),
                        'Amount': f"{amount:.1f} {'kcal' if nutrient == 'calories' else 'g'}",
                        '% Daily Value': f"{percentage:.1f}%"
                    })
                
                if nutrition_table:
                    df = pd.DataFrame(nutrition_table)
                    st.dataframe(df, width="stretch", hide_index=True)
                    
                    st.markdown("---")
                    
                    # Pie chart
                    st.markdown("### 📈 Daily Value Distribution")
                    
                    labels = [row['Nutrient'] for row in nutrition_table]
                    values = [float(row['% Daily Value'].replace('%', '')) for row in nutrition_table]
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.3
                    )])
                    fig.update_layout(
                        title=f"Percentage of Daily Values ({standard} Standard)",
                        height=400
                    )
                    st.plotly_chart(fig, width="stretch")
                    
                    # Warnings
                    st.markdown("---")
                    st.markdown("### ⚠️ Health Insights")
                    
                    for nutrient, value in nutrition_data.items():
                        amount = value * multiplier
                        daily_value = daily_values.get(nutrient, 100)
                        percentage = (amount / daily_value) * 100
                        
                        if nutrient in ['saturated_fat', 'sugars', 'salt']:
                            if percentage > 50:
                                st.error(f"🔴 High in {nutrient.replace('_', ' ').title()}: {percentage:.0f}% of daily value")
                            elif percentage > 30:
                                st.warning(f"🟡 Moderate {nutrient.replace('_', ' ').title()}: {percentage:.0f}% of daily value")
                        
                        if nutrient == 'fiber':
                            if percentage > 20:
                                st.success(f"🟢 Good source of fiber: {percentage:.0f}% of daily value")
                        
                        if nutrient == 'protein':
                            if percentage > 30:
                                st.success(f"🟢 High protein: {percentage:.0f}% of daily value")
                else:
                    st.warning("No nutrition facts detected in the image")
        
        elif uploaded_file is not None:
            with col2:
                st.info("👆 Click 'Calculate Nutrition' to analyze the image")

# ============================================================================
# PAGE: ALLERGEN ALERTS
# ============================================================================

elif selected == "Allergen Alerts":
    st.markdown("<h1 class='main-header'>🚨 Allergen Alerts</h1>", unsafe_allow_html=True)
    
    # Allergen selection
    st.markdown("### 🔍 Select Your Allergens")
    
    allergens_list = [
        'gluten', 'crustaceans', 'eggs', 'fish', 'peanuts', 'soybeans',
        'milk', 'nuts', 'celery', 'mustard', 'sesame', 'sulphites', 'lupin', 'molluscs'
    ]
    
    selected_allergens = st.multiselect(
        "I am allergic to:",
        allergens_list,
        default=[],
        help="Select all allergens you want to avoid"
    )
    
    if selected_allergens:
        st.info(f"🔍 Checking for: **{', '.join(selected_allergens)}**")
    
    st.markdown("---")
    
    # Image upload
    st.markdown("### 📸 Upload Product Image")
    
    uploaded_file = st.file_uploader(
        "Upload product image showing ingredients",
        type=['png', 'jpg', 'jpeg'],
        help="Upload image with visible ingredient list",
        key="allergen_upload"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Product Image", width="stretch")
            
            if st.button("🔍 Check for Allergens", type="primary", width="stretch"):
                with st.spinner("Scanning for allergens..."):
                    try:
                        # Import OCR
                        import easyocr
                        
                        # Initialize OCR
                        reader = easyocr.Reader(['en', 'de', 'fr', 'es'], gpu=torch.cuda.is_available(), verbose=False)
                        
                        # Extract text
                        ocr_results = reader.readtext(np.array(image))
                        ocr_text = ' '.join([text for _, text, _ in ocr_results])
                        
                        # Allergen detection patterns
                        allergen_keywords = {
                            'gluten': ['gluten', 'wheat', 'weizen', 'blé', 'trigo', 'barley', 'rye', 'oats'],
                            'crustaceans': ['crustacean', 'crab', 'lobster', 'shrimp', 'prawn', 'crevette', 'krebstier'],
                            'eggs': ['egg', 'ei', 'oeuf', 'huevo'],
                            'fish': ['fish', 'fisch', 'poisson', 'pescado'],
                            'peanuts': ['peanut', 'erdnuss', 'arachide', 'cacahuete'],
                            'soybeans': ['soy', 'soja', 'soybean'],
                            'milk': ['milk', 'milch', 'lait', 'leche', 'dairy', 'lactose', 'whey', 'casein'],
                            'nuts': ['almond', 'hazelnut', 'walnut', 'cashew', 'pecan', 'brazil nut', 'pistachio', 'macadamia', 'nuss', 'noix'],
                            'celery': ['celery', 'sellerie', 'céleri', 'apio'],
                            'mustard': ['mustard', 'senf', 'moutarde', 'mostaza'],
                            'sesame': ['sesame', 'sesam', 'sésame', 'sésamo'],
                            'sulphites': ['sulphite', 'sulfite', 'sulfit', 'sulfito', 'so2'],
                            'lupin': ['lupin', 'lupine'],
                            'molluscs': ['mollusc', 'mollusk', 'mussel', 'oyster', 'clam', 'weichtier', 'mollusque']
                        }
                        
                        # Warning patterns
                        warning_patterns = {
                            'contains': r'(contains?|enthält|contient|contiene)[:\\s]*([^.]+)',
                            'may_contain': r'(may contain|kann enthalten|peut contenir|puede contener)[:\\s]*([^.]+)'
                        }
                        
                        text_lower = ocr_text.lower()
                        detected_allergens = []
                        
                        # Check for explicit warnings
                        for warning_type, pattern in warning_patterns.items():
                            match = re.search(pattern, text_lower, re.IGNORECASE)
                            if match:
                                warning_text = match.group(2)
                                
                                for allergen_name, keywords in allergen_keywords.items():
                                    for keyword in keywords:
                                        if keyword in warning_text:
                                            detected_allergens.append({
                                                'allergen': allergen_name,
                                                'type': warning_type,
                                                'confidence': 0.9
                                            })
                                            break
                        
                        # Check general text
                        for allergen_name, keywords in allergen_keywords.items():
                            if any(d['allergen'] == allergen_name for d in detected_allergens):
                                continue
                            
                            for keyword in keywords:
                                if keyword in text_lower:
                                    detected_allergens.append({
                                        'allergen': allergen_name,
                                        'type': 'possible',
                                        'confidence': 0.6
                                    })
                                    break
                        
                        # Store results
                        st.session_state['allergen_results'] = {
                            'detected': detected_allergens,
                            'ocr_text': ocr_text,
                            'user_allergens': selected_allergens
                        }
                        
                    except Exception as e:
                        st.error(f"Error detecting allergens: {e}")
        
        # Display results
        if 'allergen_results' in st.session_state:
            results = st.session_state['allergen_results']
            detected = results['detected']
            user_allergens = results.get('user_allergens', selected_allergens)
            
            with col2:
                st.markdown("### 🎯 Allergen Check Results")
                
                # Check against user allergens
                warnings = []
                may_contain = []
                safe_allergens = []
                
                if user_allergens:
                    for allergen in detected:
                        if allergen['allergen'] in user_allergens:
                            if allergen['type'] == 'contains':
                                warnings.append(allergen)
                            elif allergen['type'] == 'may_contain':
                                may_contain.append(allergen)
                    
                    # Overall safety
                    is_safe = len(warnings) == 0
                    
                    if is_safe and len(may_contain) == 0:
                        st.success("✅ **SAFE** - No allergens detected from your list!")
                    elif is_safe:
                        st.warning("⚠️ **CAUTION** - May contain traces")
                    else:
                        st.error("⛔ **UNSAFE** - Contains allergens from your list!")
                    
                    st.markdown("---")
                    
                    # Show warnings
                    if warnings:
                        st.markdown("#### 🔴 Contains:")
                        for warning in warnings:
                            st.error(f"**{warning['allergen'].upper()}** (Confidence: {warning['confidence']:.0%})")
                    
                    if may_contain:
                        st.markdown("#### 🟡 May Contain:")
                        for warning in may_contain:
                            st.warning(f"**{warning['allergen'].title()}** (Confidence: {warning['confidence']:.0%})")
                    
                    # Safe allergens from user list
                    safe_user_allergens = [a for a in user_allergens if a not in [w['allergen'] for w in warnings + may_contain]]
                    if safe_user_allergens:
                        st.markdown("#### 🟢 Safe (from your list):")
                        for allergen in safe_user_allergens:
                            st.success(f"✅ {allergen.title()}")
                
                else:
                    st.info("👆 Select your allergens above to get personalized results")
                
                st.markdown("---")
                
                # All detected allergens
                if detected:
                    st.markdown("#### 📋 All Detected Allergens")
                    
                    allergen_summary = {}
                    for allergen in detected:
                        name = allergen['allergen']
                        if name not in allergen_summary:
                            allergen_summary[name] = allergen['type']
                    
                    for name, type_ in allergen_summary.items():
                        icon = "⚠️" if type_ in ['contains', 'may_contain'] else "ℹ️"
                        st.write(f"{icon} {name.title()} ({type_.replace('_', ' ')})")
                else:
                    st.info("No allergens detected in the image")
                
                # Show extracted text
                with st.expander("📄 View Extracted Text"):
                    st.text_area("OCR Results", results['ocr_text'], height=150)
        
        elif uploaded_file is not None:
            with col2:
                st.info("👆 Click 'Check for Allergens' to scan the image")

# ============================================================================
# PAGE: RECOMMENDATIONS
# ============================================================================

elif selected == "Recommendations":
    st.markdown("<h1 class='main-header'>⭐ Product Recommendations</h1>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📸 Analyze & Get Alternatives", "🔍 Browse by Preferences"])
    
    with tab1:
        st.markdown("### Find Healthier Alternatives")
        st.info("📸 Upload a product image to find healthier alternatives")
        
        uploaded_file = st.file_uploader(
            "Upload Product Image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload image of a product you want alternatives for",
            key="rec_upload"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Your Product", width="stretch")
                
                num_alternatives = st.slider("Number of alternatives", 1, 10, 5)
                
                if st.button("🔍 Find Healthier Alternatives", type="primary", width="stretch"):
                    with st.spinner("Analyzing product and finding alternatives..."):
                        # YOLO Detection
                        yolo_results = yolo_model(np.array(image))
                        detections = []
                        
                        for pred in yolo_results.pred[0]:
                            x1, y1, x2, y2, conf, cls = pred.tolist()
                            label_names = {
                                0: 'NutriScore_A', 1: 'NutriScore_B', 2: 'NutriScore_C',
                                3: 'NutriScore_D', 4: 'NutriScore_E', 5: 'BIO', 6: 'V-Label'
                            }
                            detections.append({
                                'label': label_names[int(cls)],
                                'confidence': float(conf)
                            })
                        
                        # Calculate health score
                        nutriscore = None
                        is_organic = False
                        is_vegan = False
                        health_score = 50.0
                        
                        for det in detections:
                            if 'NutriScore' in det['label']:
                                nutriscore = det['label']
                                score_map = {
                                    'NutriScore_A': 100, 'NutriScore_B': 80, 'NutriScore_C': 60,
                                    'NutriScore_D': 40, 'NutriScore_E': 20
                                }
                                health_score = score_map.get(nutriscore, 50)
                            elif det['label'] == 'BIO':
                                is_organic = True
                            elif det['label'] == 'V-Label':
                                is_vegan = True
                        
                        # Store current product info
                        st.session_state['current_product'] = {
                            'image': image,
                            'filename': uploaded_file.name,
                            'health_score': health_score,
                            'nutriscore': nutriscore,
                            'is_organic': is_organic,
                            'is_vegan': is_vegan
                        }
                        
                        # Find alternatives from database
                        alternatives = db.search_products(
                            min_health_score=health_score + 5,  # Better than current
                            limit=num_alternatives * 3
                        )
                        
                        # Filter and sort
                        if is_organic:
                            alternatives = [p for p in alternatives if p.get('is_organic')]
                        if is_vegan:
                            alternatives = [p for p in alternatives if p.get('is_vegan_vegetarian')]
                        
                        # Sort by health score
                        alternatives = sorted(alternatives, key=lambda x: x.get('health_score', 0), reverse=True)[:num_alternatives]
                        
                        st.session_state['alternatives'] = alternatives
            
            # Display results
            if 'current_product' in st.session_state and 'alternatives' in st.session_state:
                current = st.session_state['current_product']
                alternatives = st.session_state['alternatives']
                
                with col2:
                    st.markdown("### 📊 Your Product")
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Health Score", f"{current['health_score']:.0f}/100")
                    with col_b:
                        if current['nutriscore']:
                            st.metric("NutriScore", current['nutriscore'].replace('NutriScore_', ''))
                        else:
                            st.metric("NutriScore", "N/A")
                    with col_c:
                        badges = []
                        if current['is_organic']:
                            badges.append("🌱")
                        if current['is_vegan']:
                            badges.append("🌿")
                        st.metric("Labels", " ".join(badges) if badges else "None")
                
                st.markdown("---")
                
                if alternatives:
                    st.markdown(f"### ✅ Found {len(alternatives)} Healthier Alternatives")
                    
                    for i, alt in enumerate(alternatives, 1):
                        with st.container():
                            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                            
                            with col1:
                                st.markdown(f"**{i}. {alt.get('product_name', 'Unknown')}**")
                                st.caption(f"Brand: {alt.get('brand', 'Unknown')}")
                                
                                badges = []
                                if alt.get('is_organic'):
                                    badges.append("🌱 Organic")
                                if alt.get('is_vegan_vegetarian'):
                                    badges.append("🌿 Vegan")
                                if badges:
                                    st.write(" | ".join(badges))
                            
                            with col2:
                                improvement = alt.get('health_score', 0) - current['health_score']
                                st.metric("Health Score", f"{alt.get('health_score', 0):.0f}", f"+{improvement:.0f}")
                            
                            with col3:
                                if alt.get('nutriscore'):
                                    st.success(alt['nutriscore'].replace('NutriScore_', ''))
                                else:
                                    st.info("N/A")
                            
                            with col4:
                                # Show improvement percentage
                                if current['health_score'] > 0:
                                    improvement_pct = ((alt.get('health_score', 0) - current['health_score']) / current['health_score']) * 100
                                    if improvement_pct > 0:
                                        st.success(f"+{improvement_pct:.0f}%")
                            
                            st.markdown("---")
                else:
                    st.info("😊 Great choice! No healthier alternatives found - your product is already excellent!")
    
    with tab2:
        st.markdown("### Get Personalized Recommendations")
        st.info("🔍 Browse products by your preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            prefer_organic = st.checkbox("🌱 Prefer Organic", value=False)
            prefer_vegan = st.checkbox("🌿 Prefer Vegan/Vegetarian", value=False)
        
        with col2:
            min_score = st.slider("Minimum Health Score", 0, 100, 60)
            num_recommendations = st.slider("Number of recommendations", 1, 20, 10, key="num_rec")
        
        if st.button("Get Recommendations", type="primary"):
            recommendations = db.search_products(
                is_organic=prefer_organic if prefer_organic else None,
                is_vegan=prefer_vegan if prefer_vegan else None,
                min_health_score=min_score,
                limit=num_recommendations
            )
            
            if recommendations:
                st.markdown(f"### ⭐ {len(recommendations)} Products Recommended")
                
                # Display in grid
                for i in range(0, len(recommendations), 2):
                    cols = st.columns(2)
                    
                    for j, col in enumerate(cols):
                        if i + j < len(recommendations):
                            rec = recommendations[i + j]
                            
                            with col:
                                with st.container():
                                    st.markdown(f"**{rec.get('product_name', 'Unknown')}**")
                                    st.caption(f"Brand: {rec.get('brand', 'Unknown')}")
                                    
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        st.metric("Health Score", f"{rec.get('health_score', 0):.0f}")
                                    with col_b:
                                        if rec.get('nutriscore'):
                                            st.success(rec['nutriscore'].replace('NutriScore_', ''))
                                    
                                    badges = []
                                    if rec.get('is_organic'):
                                        badges.append("🌱 Organic")
                                    if rec.get('is_vegan_vegetarian'):
                                        badges.append("🌿 Vegan")
                                    if badges:
                                        st.write(" | ".join(badges))
                                    
                                    st.markdown("---")
            else:
                st.info("No products found matching your preferences. Try adjusting the filters.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>NutriGreen © 2024 | Powered by AI & Computer Vision</p>",
    unsafe_allow_html=True
)
