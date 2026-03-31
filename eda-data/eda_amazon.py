import pandas as pd
import numpy as np
import os

# Paths
DATA_DIR = "/Users/thiennguyen/Library/CloudStorage/GoogleDrive-nguyenvmthien@gmail.com/My Drive/A+SCHOOL-HK11/applied-big-data/Lab/Lab02/src/data/amazon-book"

def load_amazon_data(file_path):
    users = []
    items = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            user_id = int(parts[0])
            item_ids = [int(x) for x in parts[1:]]
            for item_id in item_ids:
                users.append(user_id)
                items.append(item_id)
    return pd.DataFrame({'user_id': users, 'item_id': items})

def perform_eda():
    print("--- Amazon-Book EDA ---")
    
    train_df = load_amazon_data(os.path.join(DATA_DIR, 'train.txt'))
    test_df = load_amazon_data(os.path.join(DATA_DIR, 'test.txt'))
    
    all_df = pd.concat([train_df, test_df])
    
    num_users = all_df['user_id'].nunique()
    num_items = all_df['item_id'].nunique()
    num_interactions = len(all_df)
    
    print(f"Number of users: {num_users}")
    print(f"Number of items: {num_items}")
    print(f"Number of interactions: {num_interactions}")
    
    # Sparsity
    sparsity = 1.0 - (num_interactions / (num_users * num_items))
    print(f"Sparsity: {sparsity:.6f}")
    
    # Interactions per user
    user_counts = all_df.groupby('user_id').size()
    print("\nInteractions per User Statistics:")
    print(user_counts.describe())
    
    # Interactions per item
    item_counts = all_df.groupby('item_id').size()
    print("\nInteractions per Item Statistics:")
    print(item_counts.describe())
    
    # Train/Test Split Ratio
    print(f"\nTrain interactions: {len(train_df)}")
    print(f"Test interactions: {len(test_df)}")
    print(f"Train/Test Ratio: {len(train_df)/num_interactions:.2%}")

if __name__ == "__main__":
    perform_eda()
