import os
import logging
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
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
    
    def train_on_amazon_reviews(self, dataset_path, output_dir, epochs=3, batch_size=4, learning_rate=5e-5, val_split=0.1):
        """
        Train the model on Amazon review data.
        
        Args:
            dataset_path: Path to the processed dataset CSV
            output_dir: Directory to save the trained model
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            val_split: Proportion of data to use for validation
        """
        logger.info(f"Starting training on dataset from {dataset_path}")
        
        # Load dataset
        try:
            df = pd.read_csv(dataset_path)
            logger.info(f"Loaded dataset with {len(df)} examples")
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return
        
        # Dataset class for Amazon reviews
        class ReviewDataset(Dataset):
            def __init__(self, dataframe, tokenizer, max_length=512):
                self.data = dataframe
                self.tokenizer = tokenizer
                self.max_length = max_length
                
            def __len__(self):
                return len(self.data)
                
            def __getitem__(self, idx):
                if 'prompt' in self.data.columns and 'completion' in self.data.columns:
                    # Format for prompt-completion pairs
                    prompt = self.data.iloc[idx]['prompt']
                    completion = self.data.iloc[idx]['completion']
                    text = f"{prompt}\n\n{completion}"
                else:
                    # Format for product reviews
                    product = self.data.iloc[idx].get('ProductId', '')
                    score = self.data.iloc[idx].get('Score', '')
                    summary = self.data.iloc[idx].get('Summary', '')
                    review_text = self.data.iloc[idx].get('Text', '')
                    text = f"Product: {product}\nRating: {score}/5\n{summary}\n{review_text}"
                
                # Tokenize the text
                encodings = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                # For causal language modeling, labels are the same as input_ids
                return {
                    'input_ids': encodings['input_ids'].squeeze(),
                    'attention_mask': encodings['attention_mask'].squeeze(),
                    'labels': encodings['input_ids'].squeeze()
                }
        
        # Create full dataset
        full_dataset = ReviewDataset(df, self.tokenizer)
        
        # Split into train and validation sets
        val_size = int(val_split * len(full_dataset))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )
        
        logger.info(f"Train set: {train_size} examples")
        logger.info(f"Validation set: {val_size} examples")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size
        )
        
        # Set up optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Training loop
        logger.info("Starting training...")
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            # Training phase
            self.model.train()
            train_loss = 0
            train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
            
            for batch in train_progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                train_loss += loss.item()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Update progress bar
                train_progress_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            logger.info(f"Average training loss: {avg_train_loss:.4f}")
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]")
            
            with torch.no_grad():
                for batch in val_progress_bar:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    val_loss += loss.item()
                    
                    val_progress_bar.set_postfix({'loss': loss.item()})
            
            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"Average validation loss: {avg_val_loss:.4f}")
            
            # Save the model if validation loss improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                logger.info(f"Validation loss improved. Saving model to {output_dir}")
                
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Save model and tokenizer
                self.model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
        
        logger.info("Training complete!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Model saved to {output_dir}")

    def process_amazon_reviews(self, input_file, output_file):
        """
        Process Amazon reviews dataset for training.
        
        Args:
            input_file: Path to the input CSV file
            output_file: Path to save the processed CSV file
        """
        logger.info(f"Processing Amazon reviews from {input_file}")
        
        try:
            # Load the dataset
            df = pd.read_csv(input_file)
            logger.info(f"Loaded dataset with {len(df)} rows")
            
            # Select relevant columns (adjust based on actual column names)
            necessary_columns = ['ProductId', 'Score', 'Summary']
            optional_columns = ['Text', 'review_body', 'HelpfulnessNumerator', 'HelpfulnessDenominator']
            
            # Check which columns are available
            available_columns = [col for col in df.columns if col in necessary_columns + optional_columns]
            
            # If 'Text' column doesn't exist but 'review_body' does, rename it
            if 'Text' not in available_columns and 'review_body' in available_columns:
                df = df.rename(columns={'review_body': 'Text'})
                available_columns = [col if col != 'review_body' else 'Text' for col in available_columns]
            
            # Keep only available columns
            df = df[available_columns]
            
            # Drop rows with missing data in necessary columns
            for col in necessary_columns:
                if col in available_columns:
                    df = df.dropna(subset=[col])
            
            # Handle text columns - clean up newlines, etc.
            text_columns = ['Summary', 'Text']
            for col in text_columns:
                if col in available_columns:
                    df[col] = df[col].astype(str)
                    df[col] = df[col].str.replace('\n', ' ').str.replace('\r', ' ')
                    df[col] = df[col].str.replace('\s+', ' ', regex=True)
            
            # Format for training (create prompt-completion pairs)
            df['prompt'] = "Generate a product review for this product: " + df['ProductId'].astype(str)
            
            # Create completion text
            df['completion'] = ""
            if 'Score' in available_columns:
                df['completion'] += "Rating: " + df['Score'].astype(str) + "/5\n"
            if 'Summary' in available_columns:
                df['completion'] += df['Summary'] + "\n"
            if 'Text' in available_columns:
                df['completion'] += df['Text']
            
            # Save the processed dataset
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save only prompt-completion pairs for training
            train_data = df[['prompt', 'completion']]
            train_data.to_csv(output_file, index=False)
            
            logger.info(f"Saved processed dataset to {output_file}")
            logger.info(f"Processed dataset contains {len(df)} examples")
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or use a product review generator")
    
    # Command arguments
    parser.add_argument('--mode', type=str, default='generate', choices=['generate', 'train', 'process'],
                        help='Operation mode: generate, train, or process')
    
    # Generation arguments
    parser.add_argument('--prompt', type=str, default="Write a review for Sony WH-1000XM4 headphones.",
                        help='Prompt for review generation')
    parser.add_argument('--model_path', type=str, default="models/review_gpt_model",
                        help='Path to the fine-tuned model')
    
    # Training arguments
    parser.add_argument('--data_path', type=str, default="scripts/reviews.csv",
                        help='Path to the training data')
    parser.add_argument('--output_dir', type=str, default="models/review_gpt_model",
                        help='Directory to save the trained model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    
    # Processing arguments
    parser.add_argument('--input_file', type=str, default="scripts/Reviews.csv",
                        help='Path to the raw Amazon reviews dataset')
    parser.add_argument('--output_file', type=str, default="scripts/reviews.csv",
                        help='Path to save the processed dataset')
    
    args = parser.parse_args()
    
    # Initialize the generator
    generator = ReviewGenerator(model_path=args.model_path)
    
    if args.mode == 'generate':
        # Generate a review
        reviews = generator.generate_review(args.prompt)
        print("Generated review:")
        print(reviews[0])
        
    elif args.mode == 'process':
        # Process the Amazon reviews dataset
        generator.process_amazon_reviews(args.input_file, args.output_file)
        
    elif args.mode == 'train':
        # Train the model
        generator.train_on_amazon_reviews(
            dataset_path=args.data_path,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )