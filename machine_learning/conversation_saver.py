"""
Conversation Manager
-------------------
Handles conversation state tracking, context management, and structured dialog flow
for the product review chatbot.
"""

import json
import time
import random
from enum import Enum
import os
from datetime import datetime

class ConversationState(Enum):
    """Enum to track the state of the conversation"""
    INITIAL = "initial"               # First contact
    PRODUCT_IDENTIFIED = "product"    # Product has been identified
    COLLECTING_OPINIONS = "opinions"  # Gathering user opinions on product
    GENERATING_REVIEW = "review"      # Ready to generate a review
    REFINING_REVIEW = "refine"        # Refining an existing review
    COMPARISON = "comparison"         # Comparing multiple products
    CLOSING = "closing"               # Ending the conversation

class ConversationManager:
    """
    Manages the conversation flow for the product review chatbot.
    Tracks context, handles state transitions, and provides relevant prompts.
    """
    
    def __init__(self, db_connection=None):
        """Initialize with optional database connection"""
        self.db = db_connection
        self.storage_dir = "data/conversations"
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Default response templates for different conversation states
        self.default_templates = {
            ConversationState.INITIAL: [
                "Hello! I can help you generate product reviews. What product are you interested in?",
                "Welcome! I'd be happy to discuss product reviews with you. What product would you like to focus on?",
                "Hi there! I can generate detailed reviews for products. Which item are you looking for information about?"
            ],
            ConversationState.PRODUCT_IDENTIFIED: [
                "Great! I now know we're talking about {product_name}. What aspects of it interest you most?",
                "I see we're discussing {product_name}. What specific features or qualities would you like to know about?",
                "Thanks for specifying {product_name}. Would you like to hear about particular aspects of this product?"
            ],
            ConversationState.COLLECTING_OPINIONS: [
                "What do you think about {product_name} so far? Any specific likes or dislikes?",
                "Have you had any personal experience with {product_name}? I'd love to hear your thoughts.",
                "What's your impression of {product_name} based on what you've seen or heard?"
            ],
            ConversationState.GENERATING_REVIEW: [
                "Based on our discussion about {product_name}, here's a review I've generated:",
                "After considering the features and aspects of {product_name} we discussed, here's my review:",
                "Taking into account everything we've talked about regarding {product_name}, here's a comprehensive review:"
            ],
            ConversationState.REFINING_REVIEW: [
                "How would you like me to modify this review? Should I focus more on any particular aspects?",
                "Would you like me to adjust this review in any way? Perhaps emphasize different features?",
                "Is there anything you'd like me to change about this review? More detail on certain aspects?"
            ],
            ConversationState.COMPARISON: [
                "How does {product_name} compare to {comparison_product} in your view?",
                "Would you like me to compare {product_name} with {comparison_product} in the review?",
                "When comparing {product_name} to {comparison_product}, which aspects matter most to you?"
            ],
            ConversationState.CLOSING: [
                "I hope this review of {product_name} was helpful! Is there anything else you'd like to know?",
                "That concludes our discussion about {product_name}. Was this review useful to you?",
                "Is there anything else you'd like to know about {product_name} or any other products?"
            ]
        }
    
    def get_conversation(self, user_id, conversation_id=None):
        """Retrieve the conversation history for a user"""
        if self.db:
            # Get from database
            return self.db.get_conversation(user_id, conversation_id)
        else:
            # Get from file-based storage
            filename = f"{self.storage_dir}/{user_id}_{conversation_id or 'latest'}.json"
            try:
                with open(filename, "r") as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                # Return a new conversation if none exists
                return {
                    "conversation_id": conversation_id or f"conv_{int(time.time())}",
                    "user_id": user_id,
                    "messages": [],
                    "state": ConversationState.INITIAL.value,
                    "context": {
                        "product_name": None,
                        "comparison_product": None,
                        "identified_features": [],
                        "user_opinions": {},
                        "generated_reviews": []
                    },
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
    
    def save_conversation(self, user_id, conversation):
        """Save the updated conversation"""
        conversation["updated_at"] = datetime.now().isoformat()
        
        if self.db:
            # Save to database
            self.db.save_conversation(user_id, conversation)
        else:
            # Save to file
            conversation_id = conversation.get("conversation_id", f"conv_{int(time.time())}")
            filename = f"{self.storage_dir}/{user_id}_{conversation_id}.json"
            
            with open(filename, "w") as f:
                json.dump(conversation, f, indent=2)
            
            # Also save as latest
            latest_filename = f"{self.storage_dir}/{user_id}_latest.json"
            with open(latest_filename, "w") as f:
                json.dump(conversation, f, indent=2)
    
    def add_message(self, user_id, role, content, conversation_id=None, metadata=None):
        """Add a message to the conversation history"""
        conversation = self.get_conversation(user_id, conversation_id)
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            message["metadata"] = metadata
            
        conversation["messages"].append(message)
        self.save_conversation(user_id, conversation)
        return conversation
    
    def update_state(self, user_id, new_state, context_updates=None, conversation_id=None):
        """Update the conversation state and context"""
        conversation = self.get_conversation(user_id, conversation_id)
        conversation["state"] = new_state.value
        
        if context_updates:
            for key, value in context_updates.items():
                conversation["context"][key] = value
        
        self.save_conversation(user_id, conversation)
        return conversation
    
    def get_response_template(self, state, context=None):
        """Get a template response based on the current state"""
        templates = self.default_templates.get(state, [])
        
        if not templates:
            return "I'm not sure what to say next. Can you help guide our conversation?"
        
        template = random.choice(templates)
        
        # Fill in template variables if context is provided
        if context:
            try:
                return template.format(**context)
            except KeyError:
                # If missing keys, return the raw template
                return template
        
        return template
    
    def extract_product_from_message(self, message, image_description=None):
        """
        Extract potential product mentions from user message
        This is a simplified approach - in production, use NER
        """
        # Basic keyword detection for products
        potential_products = []
        
        # If we have an image description, it likely contains the product
        if image_description:
            # Extract nouns from image description as potential products
            words = image_description.split()
            for word in words:
                if len(word) > 3 and word[0].isupper():  # Basic heuristic for product names
                    potential_products.append(word)
        
        # Look for indicators like "review for [product]" in the message
        product_indicators = [
            "review of", "review for", "reviews of", "reviews for",
            "looking at", "interested in", "thinking of buying", 
            "want to know about", "information on"
        ]
        
        message_lower = message.lower()
        for indicator in product_indicators:
            if indicator in message_lower:
                # Get text after the indicator
                index = message_lower.find(indicator) + len(indicator)
                potential_product = message[index:].strip().split(".")[0].strip()
                if potential_product:
                    potential_products.append(potential_product)
        
        if potential_products:
            # Take the longest potential product name as it's likely the most specific
            return max(potential_products, key=len)
        
        return None
    
    def extract_user_opinions(self, message, product_name):
        """
        Extract user opinions about the product from their message
        This is a simplified approach - in production, use sentiment analysis
        """
        opinions = {}
        
        opinion_indicators = {
            'like': 'positive',
            'love': 'positive',
            'great': 'positive',
            'excellent': 'positive',
            'amazing': 'positive',
            'good': 'positive',
            'awesome': 'positive',
            'dislike': 'negative',
            'hate': 'negative',
            'terrible': 'negative',
            'awful': 'negative',
            'bad': 'negative',
            'poor': 'negative',
            'disappointing': 'negative'
        }
        
        message_lower = message.lower()
        product_lower = product_name.lower() if product_name else ""
        
        # Check if message contains product name or refers to it
        if product_lower and product_lower in message_lower:
            # Extract opinion phrases around product name
            # Find opinion indicators near the product name
            for indicator, sentiment in opinion_indicators.items():
                if indicator in message_lower:
                    # Extract the surrounding context
                    start = max(0, message_lower.find(indicator) - 50)
                    end = min(len(message_lower), message_lower.find(indicator) + 50)
                    context = message_lower[start:end]
                    
                    # Try to identify what aspect of the product they're talking about
                    aspects = ["price", "quality", "design", "features", "performance", "size", 
                               "weight", "battery", "screen", "camera", "sound", "customer service"]
                    
                    found_aspect = "general"
                    for aspect in aspects:
                        if aspect in context:
                            found_aspect = aspect
                            break
                    
                    opinions[found_aspect] = {
                        "sentiment": sentiment,
                        "indicator": indicator,
                        "context": context
                    }
        
        return opinions
    
    def analyze_user_message(self, message, image_description=None):
        """
        Analyze user message to determine intent and extract entities.
        In a full implementation, this would use NLP/NLU techniques.
        """
        # This is a simplified version - in production, use a proper NLU system
        intent = {
            "action": None,
            "entities": []
        }
        
        # Very basic intent detection
        message_lower = message.lower()
        
        # Check for product identification intent
        product_indicators = ["looking for", "interested in", "want to know about", "review of", "reviews for"]
        if any(indicator in message_lower for indicator in product_indicators):
            intent["action"] = "identify_product"
        
        # Check for opinion sharing intent
        opinion_indicators = ["i think", "in my opinion", "i like", "i don't like", "i love", "i hate"]
        if any(indicator in message_lower for indicator in opinion_indicators):
            intent["action"] = "share_opinion"
        
        # Check for review request intent
        review_indicators = ["generate review", "write review", "create review", "review this"]
        if any(indicator in message_lower for indicator in review_indicators):
            intent["action"] = "generate_review"
        
        # Extract potential product names (very simplistic approach)
        # In a real implementation, use named entity recognition
        product_name = self.extract_product_from_message(message, image_description)
        if product_name:
            intent["entities"].append({
                "type": "product",
                "value": product_name,
                "confidence": 0.7
            })
        
        return intent
    
    def determine_next_state(self, current_state, user_intent, context):
        """
        Determine the next conversation state based on the current state,
        user intent, and context.
        """
        if current_state == ConversationState.INITIAL:
            if user_intent["action"] == "identify_product" and len(user_intent["entities"]) > 0:
                return ConversationState.PRODUCT_IDENTIFIED
            return ConversationState.INITIAL
            
        elif current_state == ConversationState.PRODUCT_IDENTIFIED:
            if user_intent["action"] == "share_opinion":
                return ConversationState.COLLECTING_OPINIONS
            elif user_intent["action"] == "generate_review":
                return ConversationState.GENERATING_REVIEW
            return ConversationState.PRODUCT_IDENTIFIED
            
        elif current_state == ConversationState.COLLECTING_OPINIONS:
            if user_intent["action"] == "generate_review":
                return ConversationState.GENERATING_REVIEW
            return ConversationState.COLLECTING_OPINIONS
            
        elif current_state == ConversationState.GENERATING_REVIEW:
            # After generating a review, move to refining
            return ConversationState.REFINING_REVIEW
            
        elif current_state == ConversationState.REFINING_REVIEW:
            if "comparison" in user_intent:
                return ConversationState.COMPARISON
            elif "done" in user_intent or "satisfied" in user_intent:
                return ConversationState.CLOSING
            return ConversationState.REFINING_REVIEW
            
        elif current_state == ConversationState.COMPARISON:
            if "generate" in user_intent:
                return ConversationState.GENERATING_REVIEW
            return ConversationState.COMPARISON
            
        elif current_state == ConversationState.CLOSING:
            if "new product" in user_intent:
                return ConversationState.INITIAL
            return ConversationState.CLOSING
            
        # Default: stay in current state
        return current_state
    
    def generate_prompt_from_context(self, conversation):
        """
        Generate a model prompt based on the conversation context
        This prompt will be sent to the language model to generate responses
        """
        context = conversation["context"]
        state = ConversationState(conversation["state"])
        messages = conversation["messages"][-5:]  # Get last 5 messages for context
        
        # Start with system instruction to set the tone
        prompt = "You are a helpful product review assistant that can generate detailed, honest reviews. "
        
        # Add conversation context
        if context.get("product_name"):
            prompt += f"The current product being discussed is {context['product_name']}. "
        
        if context.get("user_opinions"):
            prompt += "The user has shared these opinions: "
            for feature, opinion in context["user_opinions"].items():
                prompt += f"{feature}: {opinion.get('sentiment', 'neutral')}. "
        
        # Add recent conversation history
        prompt += "\nRecent conversation:\n"
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            prompt += f"{role}: {msg['content']}\n"
        
        # Add specific instruction based on state
        if state == ConversationState.GENERATING_REVIEW:
            prompt += "\nBased on the conversation above, generate a detailed, authentic-sounding product review that a real customer might write."
        elif state == ConversationState.COLLECTING_OPINIONS:
            prompt += "\nAsk thoughtful follow-up questions about the product to gather more user opinions."
        elif state == ConversationState.REFINING_REVIEW:
            prompt += "\nSuggest specific ways the review could be improved or modified based on user feedback."
        
        return prompt