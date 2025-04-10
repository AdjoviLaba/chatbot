"""
Generative Text Model
--------------------
Handles text generation for product reviews based on conversation context.
Uses a fine-tuned language model to generate human-like reviews.
"""

import os
import logging
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReviewGenerator:
    """A class to generate product reviews using a fine-tuned language model."""
    
    def __init__(
        self,
        model_path: str = "models/review_gpt_model",
        model_type: str = "gpt2",
        device: Optional[str] = None,
        max_length: int = 500,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
    ):
        """
        Initialize the review generator.
        
        Args:
            model_path: Path to the fine-tuned model directory
            model_type: Type of model to use (default: gpt2)
            device: Device to run the model on (default: CUDA if available, else CPU)
            max_length: Maximum length of generated reviews
            temperature: Sampling temperature (higher = more creative)
            top_k: Sample from top k tokens
            top_p: Sample from tokens with cumulative probability >= top_p
        """
        self.model_path = model_path
        self.model_type = model_type
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        
        # Set device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing ReviewGenerator with model path: {model_path}")
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            # Try to load the fine-tuned model
            if os.path.exists(self.model_path):
                logger.info(f"Loading fine-tuned model from {self.model_path}")
                self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
                self.model = GPT2LMHeadModel.from_pretrained(self.model_path).to(self.device)
            else:
                # Fall back to pre-trained model
                logger.warning(f"Fine-tuned model not found at {self.model_path}. Using pre-trained model {self.model_type}")
                self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_type)
                self.model = GPT2LMHeadModel.from_pretrained(self.model_type).to(self.device)
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def generate_review(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generate a product review based on the provided prompt.
        
        Args:
            prompt: Text prompt to generate from
            max_length: Maximum length of generated text (overrides instance value)
            temperature: Sampling temperature (overrides instance value)
            top_k: Top-k sampling parameter (overrides instance value)
            top_p: Top-p sampling parameter (overrides instance value)
            num_return_sequences: Number of alternative reviews to generate
            
        Returns:
            List of generated review texts
        """
        # Use instance values as defaults
        max_length = max_length or self.max_length
        temperature = temperature or self.temperature
        top_k = top_k or self.top_k
        top_p = top_p or self.top_p
        
        try:
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Set the generation parameters
            gen_kwargs = {
                "max_length": max_length,
                "num_return_sequences": num_return_sequences,
                "do_sample": True,
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
                "pad_token_id": self.tokenizer.eos_token_id,
                "attention_mask": inputs.get("attention_mask", None)
            }
            
            # Generate text
            output_sequences = self.model.generate(
                input_ids=inputs["input_ids"],
                **gen_kwargs
            )
            
            # Decode the generated text
            generated_texts = []
            for output in output_sequences:
                generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                
                # Try to extract only the generated part after the prompt
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                
                generated_texts.append(generated_text)
            
            return generated_texts
            
        except Exception as e:
            logger.error(f"Error generating review: {str(e)}")
            return [f"Error generating review: {str(e)}"]
    
    def generate_review_with_template(
        self,
        product_name: str,
        rating: Optional[int] = None,
        features: Optional[Dict[str, str]] = None,
        user_opinions: Optional[Dict[str, Dict]] = None,
        comparison_product: Optional[str] = None
    ) -> str:
        """
        Generate a review using a structured template.
        
        Args:
            product_name: Name of the product
            rating: Rating out of 5 (optional)
            features: Dictionary of product features and descriptions
            user_opinions: Dictionary of user opinions about features
            comparison_product: Name of product to compare with (optional)
            
        Returns:
            Generated review text
        """
        # Create a structured prompt
        prompt = f"Write a detailed product review for {product_name}.\n\n"
        
        if rating is not None:
            prompt += f"Overall rating: {rating}/5\n\n"
        
        if features:
            prompt += "Product features:\n"
            for feature, description in features.items():
                prompt += f"- {feature}: {description}\n"
            prompt += "\n"
        
        if user_opinions:
            prompt += "User opinions:\n"
            for feature, opinion in user_opinions.items():
                sentiment = opinion.get("sentiment", "neutral")
                prompt += f"- {feature}: {sentiment}\n"
            prompt += "\n"
        
        if comparison_product:
            prompt += f"Compare with: {comparison_product}\n\n"
        
        prompt += "Review:"
        
        # Generate the review
        reviews = self.generate_review(prompt)
        return reviews[0] if reviews else ""
    
    def generate_response_from_context(self, conversation_context: Dict) -> str:
        """
        Generate a response based on conversation context.
        
        Args:
            conversation_context: Dictionary containing conversation state and history
            
        Returns:
            Generated response text
        """
        # Extract context
        state = conversation_context.get("state", "initial")
        context = conversation_context.get("context", {})
        messages = conversation_context.get("messages", [])
        
        # Build the prompt
        prompt = "You are a helpful product review assistant. "
        
        # Add product context
        product_name = context.get("product_name")
        if product_name:
            prompt += f"The user is asking about {product_name}. "
        
        # Add conversation history
        prompt += "\nConversation history:\n"
        for msg in messages[-5:]:  # Last 5 messages
            role = "User" if msg.get("role") == "user" else "Assistant"
            prompt += f"{role}: {msg.get('content', '')}\n"
        
        # Add instruction based on state
        if state == "review":
            prompt += "\nGenerate a detailed product review based on the conversation."
        elif state == "refine":
            prompt += "\nSuggest improvements to the review based on user feedback."
        else:
            prompt += "\nRespond helpfully to the user's last message."
        
        prompt += "\nAssistant:"
        
        # Generate the response
        responses = self.generate_review(prompt)
        return responses[0] if responses else ""


# Example usage
if __name__ == "__main__":
    # Initialize the review generator
    generator = ReviewGenerator()
    
    # Generate a simple review
    prompt = "Write a review for the Sony WH-1000XM4 headphones."
    reviews = generator.generate_review(prompt)
    
    print("Generated review:")
    print(reviews[0])
    
    # Generate a more complex review with template
    review = generator.generate_review_with_template(
        product_name="Sony WH-1000XM4",
        rating=4,
        features={
            "Noise Cancellation": "Active noise cancellation technology",
            "Battery Life": "30 hours of playback time",
            "Comfort": "Cushioned ear pads and adjustable headband"
        },
        user_opinions={
            "Sound Quality": {"sentiment": "positive"},
            "Price": {"sentiment": "negative"},
            "Durability": {"sentiment": "neutral"}
        }
    )
    
    print("\nGenerated review with template:")
    print(review)