# In a new file called review_routes.py
from flask import Blueprint, request, jsonify, current_app
import re

test_bp = Blueprint('review', __name__)  

def get_prompt_for_product(product_name):
    product_lower = product_name.lower()
    
    categories = {
        "electronics": ["iphone", "phone", "smartphone", "laptop", "computer", "tablet", "headphone", "earbud", "electronics"],
        "coffee": ["coffee", "espresso", "latte", "brew", "k-cup", "bean"],
        "footwear": ["nike", "shoe", "sneaker", "boot", "running"],
        "appliance": ["maker", "machine", "blender", "toaster", "refrigerator", "microwave"]
    }
    
    # Determine product category
    detected_category = "general"
    for category, terms in categories.items():
        if any(term in product_lower for term in terms):
            detected_category = category
            break
    
    # Create category-specific prompt
    category_context = {
        "electronics": "This is an electronic device. Focus on features like battery life, screen quality, performance, and user experience.",
        "coffee": "This is a coffee product. Focus on flavor, brewing quality, value, and taste.",
        "footwear": "This is a footwear product. Focus on comfort, style, durability, and fit.",
        "appliance": "This is a home appliance. Focus on functionality, ease of use, durability, and performance."
    }.get(detected_category, "")
    
    prompt = f"""
    Summarize what people generally think about the **{product_name}** (category: {detected_category}) based on multiple reviews.
    {category_context}
    ** Requirements: **
    - Focus ONLY on aspects relevant to {detected_category}.
    - Do NOT mention unrelated categories (e.g., coffee for electronics).
    - Include:
        - Overall rating trends (average stars)
        - Common praise points
        - Common criticisms
        - Durability/longevity (if applicable)
    
    Start with "Based on reviews, the {product_name} is..." and focus ONLY on this specific product type.
    """
    
    return prompt

import logging

@test_bp.route('/generate_review', methods=['POST'])
def generate_review():
    data = request.json
    product_name = data.get('product_name', '').strip()

    if not product_name:
        return jsonify({"error": "No product specified"}), 400

    try:
        review_generator = current_app.review_generator
        prompt = get_prompt_for_product(product_name)
        current_app.logger.info(f"Generated prompt: {prompt}")  # Log the prompt

        response = review_generator.generate_review(prompt)[0]
        response = re.sub(r'<[^>]+>', '', response).strip()

        if not response:
            return jsonify({"error": "Empty response from model"}), 500

        return jsonify({"product": product_name, "review": response})
    except Exception as e:
        current_app.logger.error(f"Error generating review: {str(e)}", exc_info=True)
        return jsonify({"error": f"Generation error: {str(e)}"}), 500