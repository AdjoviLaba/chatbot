import pandas as pd
import os

def process_amazon_reviews(input_file, output_file):
    """Process Amazon reviews dataset for training."""
    print(f"Loading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Select relevant columns (adjust based on actual column names)
    columns_to_keep = ['ProductId', 'Score', 'Summary', 'Text'] 
    if 'Text' not in df.columns and 'review_body' in df.columns:
        # Rename columns if needed
        df = df.rename(columns={'review_body': 'Text'})
    
    # Keep only columns that exist in the dataset
    available_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[available_columns]
    
    # Drop rows with missing data
    df = df.dropna(subset=available_columns)
    
    # Basic text cleaning
    text_columns = ['Summary', 'Text']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].str.replace('\n', ' ').str.replace('\r', ' ')
    
    # Format for training (create prompt-completion pairs)
    df['prompt'] = "Generate a product review for this product: " + df['ProductId']
    df['completion'] = df.apply(
        lambda row: f"Rating: {row['Score']}/5\n{row['Summary']}\n{row.get('Text', '')}", 
        axis=1
    )
    
    # Save processed dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save formatted pairs for training
    train_data = df[['prompt', 'completion']]
    train_data.to_csv(output_file, index=False)
    
    print(f"Saved processed dataset to {output_file}")
    print(f"Dataset contains {len(df)} examples")

if __name__ == "__main__":
    process_amazon_reviews(
        input_file="scripts/reviews.csv",
        output_file="scripts/training.csv"
    )