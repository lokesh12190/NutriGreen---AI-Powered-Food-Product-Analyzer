"""
NutriGreen Advanced Analysis Module
Comparative analysis, nutrition calculator, allergen filter, and recommendations

All features work 100% offline with no API costs!
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
from dataclasses import dataclass


@dataclass
class DailyValues:
    """Daily recommended values"""
    calories: float = 2000  # kcal
    total_fat: float = 70  # g
    saturated_fat: float = 20  # g
    carbohydrates: float = 260  # g
    sugars: float = 90  # g
    protein: float = 50  # g
    salt: float = 6  # g
    fiber: float = 25  # g


class ProductDatabase:
    """
    Interface for querying the product database
    """
    
    def __init__(self, db_path):
        self.db_path = db_path
    
    def get_product_by_id(self, product_id: int) -> Optional[Dict]:
        """Get product details by ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM products WHERE id = ?', (product_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def get_products_by_ids(self, product_ids: List[int]) -> List[Dict]:
        """Get multiple products by IDs"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        placeholders = ','.join('?' * len(product_ids))
        cursor.execute(f'SELECT * FROM products WHERE id IN ({placeholders})', product_ids)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_nutrition_facts(self, product_id: int) -> Optional[Dict]:
        """Get nutrition facts for a product"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM nutrition_facts WHERE product_id = ?', (product_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def get_allergens(self, product_id: int) -> List[Dict]:
        """Get allergens for a product"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT allergen_name, allergen_type, confidence
            FROM allergens WHERE product_id = ?
        ''', (product_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def search_products(self, 
                       brand: Optional[str] = None,
                       nutriscore: Optional[str] = None,
                       is_organic: Optional[bool] = None,
                       is_vegan: Optional[bool] = None,
                       min_health_score: Optional[float] = None,
                       limit: int = 10) -> List[Dict]:
        """Search products with filters"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM products WHERE 1=1"
        params = []
        
        if brand:
            query += " AND brand LIKE ?"
            params.append(f"%{brand}%")
        
        if nutriscore:
            query += " AND nutriscore = ?"
            params.append(nutriscore)
        
        if is_organic is not None:
            query += " AND is_organic = ?"
            params.append(1 if is_organic else 0)
        
        if is_vegan is not None:
            query += " AND is_vegan_vegetarian = ?"
            params.append(1 if is_vegan else 0)
        
        if min_health_score is not None:
            query += " AND health_score >= ?"
            params.append(min_health_score)
        
        query += f" LIMIT {limit}"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]


class ComparativeAnalyzer:
    """
    Compare multiple products side-by-side
    """
    
    def __init__(self, database: ProductDatabase):
        self.db = database
    
    def compare_products(self, product_ids: List[int]) -> Dict:
        """
        Compare multiple products
        
        Args:
            product_ids: List of product IDs to compare
        
        Returns:
            dict: Comparison data
        """
        if len(product_ids) < 2:
            raise ValueError("Need at least 2 products to compare")
        
        if len(product_ids) > 5:
            raise ValueError("Maximum 5 products can be compared at once")
        
        # Get product details
        products = self.db.get_products_by_ids(product_ids)
        
        # Get nutrition and allergen data
        comparison_data = []
        
        for product in products:
            nutrition = self.db.get_nutrition_facts(product['id'])
            allergens = self.db.get_allergens(product['id'])
            
            comparison_data.append({
                'id': product['id'],
                'product_name': product['product_name'] or 'Unknown',
                'brand': product['brand'] or 'Unknown',
                'nutriscore': product['nutriscore'],
                'health_score': product['health_score'],
                'is_organic': bool(product['is_organic']),
                'is_vegan': bool(product['is_vegan_vegetarian']),
                'nutrition': nutrition,
                'allergens': allergens
            })
        
        # Create comparison table
        comparison_df = pd.DataFrame(comparison_data)
        
        # Rank products
        rankings = self._rank_products(comparison_data)
        
        return {
            'products': comparison_data,
            'comparison_table': comparison_df,
            'rankings': rankings,
            'winner': rankings['overall'][0] if rankings['overall'] else None
        }
    
    def _rank_products(self, products: List[Dict]) -> Dict:
        """Rank products by different criteria"""
        # Overall ranking (by health score)
        overall = sorted(products, key=lambda x: x['health_score'], reverse=True)
        
        # Nutrition ranking (if available)
        nutrition_ranked = []
        for p in products:
            if p['nutrition']:
                # Calculate nutrition score (lower calories, fat, sugar = better)
                score = 100
                if p['nutrition'].get('calories'):
                    score -= min(p['nutrition']['calories'] / 10, 40)
                if p['nutrition'].get('total_fat'):
                    score -= min(p['nutrition']['total_fat'] * 2, 30)
                if p['nutrition'].get('sugars'):
                    score -= min(p['nutrition']['sugars'] * 2, 30)
                
                nutrition_ranked.append((p['id'], score))
        
        nutrition_ranked = sorted(nutrition_ranked, key=lambda x: x[1], reverse=True)
        
        return {
            'overall': [p['id'] for p in overall],
            'nutrition': [p_id for p_id, _ in nutrition_ranked],
            'organic': [p['id'] for p in products if p['is_organic']],
            'vegan': [p['id'] for p in products if p['is_vegan']]
        }
    
    def create_comparison_chart(self, comparison_data: Dict) -> go.Figure:
        """Create interactive comparison chart"""
        products = comparison_data['products']
        
        categories = ['Health Score', 'Organic', 'Vegan', 'NutriScore']
        
        fig = go.Figure()
        
        nutriscore_map = {
            'NutriScore_A': 100,
            'NutriScore_B': 80,
            'NutriScore_C': 60,
            'NutriScore_D': 40,
            'NutriScore_E': 20,
            None: 50
        }
        
        for product in products:
            values = [
                product['health_score'],
                100 if product['is_organic'] else 0,
                100 if product['is_vegan'] else 0,
                nutriscore_map.get(product['nutriscore'], 50)
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                name=product['product_name'][:30],
                fill='toself'
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Product Comparison Radar Chart",
            showlegend=True
        )
        
        return fig


class NutritionCalculator:
    """
    Calculate nutritional information and daily intake percentages
    """
    
    def __init__(self, database: ProductDatabase):
        self.db = database
        self.eu_daily_values = DailyValues()  # EU standards
        self.us_daily_values = DailyValues(  # US FDA standards
            calories=2000,
            total_fat=78,
            saturated_fat=20,
            carbohydrates=275,
            sugars=50,
            protein=50,
            salt=5,
            fiber=28
        )
    
    def calculate_daily_values(self, 
                               product_id: int, 
                               serving_amount: float = 100,
                               standard: str = 'EU') -> Optional[Dict]:
        """
        Calculate % of daily values
        
        Args:
            product_id: Product ID
            serving_amount: Amount in grams
            standard: 'EU' or 'US'
        
        Returns:
            dict: Nutrition data with % daily values
        """
        nutrition = self.db.get_nutrition_facts(product_id)
        
        if not nutrition:
            return None
        
        daily_values = self.eu_daily_values if standard == 'EU' else self.us_daily_values
        
        # Calculate percentages (nutrition facts are per 100g)
        multiplier = serving_amount / 100
        
        result = {
            'serving_amount': serving_amount,
            'standard': standard,
            'nutrients': {}
        }
        
        nutrient_map = {
            'calories': 'calories',
            'total_fat': 'total_fat',
            'saturated_fat': 'saturated_fat',
            'carbohydrates': 'carbohydrates',
            'sugars': 'sugars',
            'protein': 'protein',
            'salt': 'salt',
            'fiber': 'fiber'
        }
        
        for db_key, dv_key in nutrient_map.items():
            if nutrition.get(db_key) is not None:
                amount = nutrition[db_key] * multiplier
                daily_value = getattr(daily_values, dv_key)
                percentage = (amount / daily_value) * 100
                
                result['nutrients'][dv_key] = {
                    'amount': round(amount, 1),
                    'unit': 'kcal' if db_key == 'calories' else 'g',
                    'daily_value': daily_value,
                    'percentage': round(percentage, 1)
                }
        
        return result
    
    def calculate_meal_nutrition(self, product_servings: List[tuple]) -> Dict:
        """
        Calculate combined nutrition for multiple products
        
        Args:
            product_servings: List of (product_id, serving_amount) tuples
        
        Returns:
            dict: Combined nutrition data
        """
        total_nutrition = {
            'calories': 0,
            'total_fat': 0,
            'saturated_fat': 0,
            'carbohydrates': 0,
            'sugars': 0,
            'protein': 0,
            'salt': 0,
            'fiber': 0
        }
        
        for product_id, serving_amount in product_servings:
            nutrition = self.db.get_nutrition_facts(product_id)
            if nutrition:
                multiplier = serving_amount / 100
                for key in total_nutrition.keys():
                    if nutrition.get(key) is not None:
                        total_nutrition[key] += nutrition[key] * multiplier
        
        # Calculate percentages
        daily_values = self.eu_daily_values
        result = {'total': total_nutrition, 'percentages': {}}
        
        for key, value in total_nutrition.items():
            daily_value = getattr(daily_values, key)
            result['percentages'][key] = round((value / daily_value) * 100, 1)
        
        return result
    
    def get_nutrition_label(self, product_id: int, serving_amount: float = 100) -> str:
        """Generate nutrition label text"""
        nutrition = self.db.get_nutrition_facts(product_id)
        
        if not nutrition:
            return "Nutrition information not available"
        
        multiplier = serving_amount / 100
        
        label = f"Nutrition Facts (per {serving_amount}g)\n"
        label += "=" * 40 + "\n"
        
        if nutrition.get('calories'):
            label += f"Calories: {nutrition['calories'] * multiplier:.0f} kcal\n"
        
        if nutrition.get('total_fat'):
            label += f"Total Fat: {nutrition['total_fat'] * multiplier:.1f}g\n"
        
        if nutrition.get('saturated_fat'):
            label += f"  Saturated Fat: {nutrition['saturated_fat'] * multiplier:.1f}g\n"
        
        if nutrition.get('carbohydrates'):
            label += f"Carbohydrates: {nutrition['carbohydrates'] * multiplier:.1f}g\n"
        
        if nutrition.get('sugars'):
            label += f"  Sugars: {nutrition['sugars'] * multiplier:.1f}g\n"
        
        if nutrition.get('fiber'):
            label += f"  Fiber: {nutrition['fiber'] * multiplier:.1f}g\n"
        
        if nutrition.get('protein'):
            label += f"Protein: {nutrition['protein'] * multiplier:.1f}g\n"
        
        if nutrition.get('salt'):
            label += f"Salt: {nutrition['salt'] * multiplier:.1f}g\n"
        
        return label


class AllergenFilter:
    """
    Filter products by allergens and create alerts
    """
    
    def __init__(self, database: ProductDatabase):
        self.db = database
    
    def filter_by_allergens(self, 
                           exclude_allergens: List[str],
                           confidence_threshold: float = 0.7) -> List[int]:
        """
        Get products that don't contain specified allergens
        
        Args:
            exclude_allergens: List of allergen names to avoid
            confidence_threshold: Minimum confidence to consider
        
        Returns:
            list: Safe product IDs
        """
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        # Get all products
        cursor.execute('SELECT id FROM products')
        all_products = [row[0] for row in cursor.fetchall()]
        
        # Get products with excluded allergens
        placeholders = ','.join('?' * len(exclude_allergens))
        cursor.execute(f'''
            SELECT DISTINCT product_id FROM allergens
            WHERE allergen_name IN ({placeholders})
            AND confidence >= ?
        ''', (*exclude_allergens, confidence_threshold))
        
        unsafe_products = set(row[0] for row in cursor.fetchall())
        conn.close()
        
        # Return safe products
        safe_products = [p for p in all_products if p not in unsafe_products]
        
        return safe_products
    
    def check_allergens(self, product_id: int, user_allergens: List[str]) -> Dict:
        """
        Check if product contains user's allergens
        
        Args:
            product_id: Product to check
            user_allergens: User's allergen list
        
        Returns:
            dict: Alert information
        """
        allergens = self.db.get_allergens(product_id)
        
        warnings = []
        may_contain = []
        
        for allergen in allergens:
            if allergen['allergen_name'] in user_allergens:
                if allergen['allergen_type'] == 'contains':
                    warnings.append({
                        'allergen': allergen['allergen_name'],
                        'severity': 'HIGH',
                        'message': f"⚠️ CONTAINS {allergen['allergen_name'].upper()}"
                    })
                elif allergen['allergen_type'] == 'may_contain':
                    may_contain.append({
                        'allergen': allergen['allergen_name'],
                        'severity': 'MEDIUM',
                        'message': f"⚠️ MAY CONTAIN {allergen['allergen_name'].upper()}"
                    })
        
        is_safe = len(warnings) == 0
        
        return {
            'safe': is_safe,
            'warnings': warnings,
            'may_contain': may_contain,
            'severity': 'HIGH' if warnings else 'MEDIUM' if may_contain else 'SAFE'
        }
    
    def get_allergen_summary(self) -> pd.DataFrame:
        """Get summary of allergens in database"""
        conn = sqlite3.connect(self.db.db_path)
        
        df = pd.read_sql_query('''
            SELECT 
                allergen_name,
                COUNT(DISTINCT product_id) as product_count,
                AVG(confidence) as avg_confidence,
                allergen_type
            FROM allergens
            GROUP BY allergen_name, allergen_type
            ORDER BY product_count DESC
        ''', conn)
        
        conn.close()
        return df


class RecommendationEngine:
    """
    Recommend healthier alternatives and similar products
    """
    
    def __init__(self, database: ProductDatabase, 
                 embeddings_index_path: str,
                 embeddings_metadata_path: str):
        self.db = database
        
        # Load embeddings
        try:
            self.index = faiss.read_index(embeddings_index_path)
            with open(embeddings_metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embeddings_available = True
            print("✅ Embeddings loaded for recommendations")
        except:
            self.embeddings_available = False
            print("⚠️ Embeddings not available - similarity search disabled")
    
    def find_healthier_alternatives(self, 
                                    product_id: int, 
                                    top_k: int = 5) -> List[Dict]:
        """
        Find healthier alternatives to a product
        
        Args:
            product_id: Product to find alternatives for
            top_k: Number of alternatives
        
        Returns:
            list: Healthier alternative products
        """
        # Get original product
        product = self.db.get_product_by_id(product_id)
        
        if not product:
            return []
        
        # Find similar products with better health scores
        conn = sqlite3.connect(self.db.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get products with better health score
        cursor.execute('''
            SELECT * FROM products
            WHERE health_score > ?
            AND id != ?
            ORDER BY health_score DESC
            LIMIT ?
        ''', (product['health_score'], product_id, top_k * 2))
        
        alternatives = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        # Filter by similar brand if possible
        same_brand = [p for p in alternatives if p['brand'] == product['brand']]
        
        if len(same_brand) >= top_k:
            return same_brand[:top_k]
        else:
            return alternatives[:top_k]
    
    def find_similar_products(self, product_id: int, top_k: int = 5) -> List[Dict]:
        """Find similar products using embeddings"""
        if not self.embeddings_available:
            return []
        
        # Find product index in metadata
        product_idx = None
        for i, meta in enumerate(self.metadata):
            if meta['id'] == product_id:
                product_idx = i
                break
        
        if product_idx is None:
            return []
        
        # Get embedding
        embedding = self.index.reconstruct(product_idx)
        
        # Search for similar
        distances, indices = self.index.search(
            embedding.reshape(1, -1).astype('float32'), 
            top_k + 1
        )
        
        # Get products (skip first as it's the same product)
        similar_products = []
        for idx, dist in zip(indices[0][1:], distances[0][1:]):
            product_id = self.metadata[idx]['id']
            product = self.db.get_product_by_id(product_id)
            if product:
                product['similarity_score'] = float(1 / (1 + dist))
                similar_products.append(product)
        
        return similar_products
    
    def recommend_by_preferences(self,
                                 prefer_organic: bool = False,
                                 prefer_vegan: bool = False,
                                 min_health_score: float = 60,
                                 exclude_allergens: List[str] = None,
                                 limit: int = 10) -> List[Dict]:
        """Recommend products based on user preferences"""
        conn = sqlite3.connect(self.db.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM products WHERE health_score >= ?"
        params = [min_health_score]
        
        if prefer_organic:
            query += " AND is_organic = 1"
        
        if prefer_vegan:
            query += " AND is_vegan_vegetarian = 1"
        
        # Exclude products with allergens
        if exclude_allergens:
            allergen_filter = AllergenFilter(self.db)
            safe_products = allergen_filter.filter_by_allergens(exclude_allergens)
            
            if safe_products:
                placeholders = ','.join('?' * len(safe_products))
                query += f" AND id IN ({placeholders})"
                params.extend(safe_products)
        
        query += " ORDER BY health_score DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        recommendations = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return recommendations