import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = "/Users/thiennguyen/Library/CloudStorage/GoogleDrive-nguyenvmthien@gmail.com/My Drive/A+SCHOOL-HK11/applied-big-data/Lab/Lab02/src/data/movielens"
OUT_DIR = os.path.join(DATA_DIR, "preprocessed")

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

def preprocess():
    # Load ratings
    ratings_cols = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    ratings = pd.read_csv(os.path.join(DATA_DIR, 'ratings.dat'), sep='::', engine='python', names=ratings_cols, encoding='ISO-8859-1')
    
    # 1. Remap UserID and MovieID
    user_map = {uid: i for i, uid in enumerate(ratings['UserID'].unique())}
    item_map = {mid: i for i, mid in enumerate(ratings['MovieID'].unique())}
    
    ratings['user_id'] = ratings['UserID'].map(user_map)
    ratings['item_id'] = ratings['MovieID'].map(item_map)
    
    # 2. Keep only relevant columns and convert to implicit
    # In NCF/ConvNCF, any rating is treated as a positive interaction (1).
    df = ratings[['user_id', 'item_id', 'Rating', 'Timestamp']].copy()
    df['Rating'] = 1  # Convert all ratings to 1 for implicit feedback
    df.rename(columns={'Rating': 'rating'}, inplace=True)
    
    # 3. Split into train and test
    # We can use a simple random split for this lab, or leave-one-out. 
    # Let's do a 80/20 split.
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['user_id'])
    
    # 4. Save mapped files
    train.to_csv(os.path.join(OUT_DIR, 'train.csv'), index=False)
    test.to_csv(os.path.join(OUT_DIR, 'test.csv'), index=False)
    
    # Save maps
    pd.DataFrame(list(user_map.items()), columns=['original_id', 'remap_id']).to_csv(os.path.join(OUT_DIR, 'user_map.csv'), index=False)
    pd.DataFrame(list(item_map.items()), columns=['original_id', 'remap_id']).to_csv(os.path.join(OUT_DIR, 'item_map.csv'), index=False)
    
    print(f"Preprocessing complete. Files saved in {OUT_DIR}")
    print(f"Train size: {len(train)}")
    print(f"Test size: {len(test)}")

if __name__ == "__main__":
    preprocess()
