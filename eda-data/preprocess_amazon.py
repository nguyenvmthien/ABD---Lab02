import pandas as pd
import numpy as np
import os

# Paths
DATA_DIR = "/Users/thiennguyen/Library/CloudStorage/GoogleDrive-nguyenvmthien@gmail.com/My Drive/A+SCHOOL-HK11/applied-big-data/Lab/Lab02/src/data/amazon-book"
OUT_DIR = os.path.join(DATA_DIR, "preprocessed")

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

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

def preprocess():
    print("Preprocessing Amazon-Book...")
    
    train_df = load_amazon_data(os.path.join(DATA_DIR, 'train.txt'))
    test_df = load_amazon_data(os.path.join(DATA_DIR, 'test.txt'))
    
    # Save as CSV
    train_df.to_csv(os.path.join(OUT_DIR, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(OUT_DIR, 'test.csv'), index=False)
    
    print(f"Preprocessing complete. Files saved in {OUT_DIR}")
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")

if __name__ == "__main__":
    preprocess()
