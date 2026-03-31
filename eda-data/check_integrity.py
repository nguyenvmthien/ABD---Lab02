import pandas as pd
import numpy as np
import os

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def check_movielens():
    print("=" * 70)
    print("KIỂM TRA TÍNH TOÀN VẸN - MOVIELENS 1M")
    print("=" * 70)

    ml = os.path.join(BASE, "movielens")

    # Load raw data
    users = pd.read_csv(os.path.join(ml, 'users.dat'), sep='::', engine='python',
                        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], encoding='ISO-8859-1')
    movies = pd.read_csv(os.path.join(ml, 'movies.dat'), sep='::', engine='python',
                         names=['MovieID', 'Title', 'Genres'], encoding='ISO-8859-1')
    ratings = pd.read_csv(os.path.join(ml, 'ratings.dat'), sep='::', engine='python',
                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'], encoding='ISO-8859-1')

    # 1. Missing values
    print("\n--- 1. Giá trị thiếu (Missing Values) ---")
    print(f"  users.dat:   {users.isnull().sum().sum()} giá trị thiếu")
    print(f"  movies.dat:  {movies.isnull().sum().sum()} giá trị thiếu")
    print(f"  ratings.dat: {ratings.isnull().sum().sum()} giá trị thiếu")

    # 2. Duplicates
    print("\n--- 2. Bản ghi trùng lặp (Duplicates) ---")
    dup_users = users.duplicated().sum()
    dup_movies = movies.duplicated().sum()
    dup_ratings = ratings.duplicated().sum()
    dup_ratings_uid = ratings.duplicated(subset=['UserID', 'MovieID']).sum()
    print(f"  users trùng lặp:   {dup_users}")
    print(f"  movies trùng lặp:  {dup_movies}")
    print(f"  ratings trùng lặp: {dup_ratings}")
    print(f"  ratings trùng lặp (UserID, MovieID): {dup_ratings_uid}")

    # 3. Data type check
    print("\n--- 3. Kiểu dữ liệu (Data Types) ---")
    print(f"  UserID dtype:  {ratings['UserID'].dtype}")
    print(f"  MovieID dtype: {ratings['MovieID'].dtype}")
    print(f"  Rating dtype:  {ratings['Rating'].dtype}")
    print(f"  Timestamp dtype: {ratings['Timestamp'].dtype}")

    # 4. Range checks
    print("\n--- 4. Kiểm tra phạm vi giá trị (Range Checks) ---")
    print(f"  UserID range: [{ratings['UserID'].min()}, {ratings['UserID'].max()}]")
    print(f"  MovieID range: [{ratings['MovieID'].min()}, {ratings['MovieID'].max()}]")
    print(f"  Rating range:  [{ratings['Rating'].min()}, {ratings['Rating'].max()}]")
    invalid_ratings = ratings[(ratings['Rating'] < 1) | (ratings['Rating'] > 5)]
    print(f"  Ratings ngoài phạm vi [1, 5]: {len(invalid_ratings)}")

    # 5. Referential integrity
    print("\n--- 5. Tính toàn vẹn tham chiếu (Referential Integrity) ---")
    users_in_ratings = set(ratings['UserID'].unique())
    users_in_users = set(users['UserID'].unique())
    orphan_users_in_ratings = users_in_ratings - users_in_users
    print(f"  UserIDs trong ratings nhưng KHÔNG có trong users.dat: {len(orphan_users_in_ratings)}")

    items_in_ratings = set(ratings['MovieID'].unique())
    items_in_movies = set(movies['MovieID'].unique())
    orphan_items_in_ratings = items_in_ratings - items_in_movies
    print(f"  MovieIDs trong ratings nhưng KHÔNG có trong movies.dat: {len(orphan_items_in_ratings)}")
    if orphan_items_in_ratings:
        print(f"    IDs: {sorted(orphan_items_in_ratings)[:10]}...")

    # 6. Preprocessed data integrity
    print("\n--- 6. Kiểm tra dữ liệu đã tiền xử lý ---")
    pp = os.path.join(ml, 'preprocessed')
    train = pd.read_csv(os.path.join(pp, 'train.csv'))
    test = pd.read_csv(os.path.join(pp, 'test.csv'))

    total_pp = len(train) + len(test)
    print(f"  Tổng bản ghi (train + test): {total_pp}")
    print(f"  Tổng bản ghi gốc:           {len(ratings)}")
    print(f"  Khớp: {'✅ CÓ' if total_pp == len(ratings) else '❌ KHÔNG'}")

    # Missing values in preprocessed
    print(f"  train.csv null: {train.isnull().sum().sum()}")
    print(f"  test.csv null:  {test.isnull().sum().sum()}")

    # ID range check
    print(f"  user_id range (preprocessed): [{min(train['user_id'].min(), test['user_id'].min())}, {max(train['user_id'].max(), test['user_id'].max())}]")
    print(f"  item_id range (preprocessed): [{min(train['item_id'].min(), test['item_id'].min())}, {max(train['item_id'].max(), test['item_id'].max())}]")

    # Check all users in train
    train_users = set(train['user_id'].unique())
    test_users = set(test['user_id'].unique())
    cold_start_users = test_users - train_users
    print(f"  Cold-start users (trong test nhưng không trong train): {len(cold_start_users)}")


def check_amazon():
    print("\n" + "=" * 70)
    print("KIỂM TRA TÍNH TOÀN VẸN - AMAZON-BOOK")
    print("=" * 70)

    ab = os.path.join(BASE, "amazon-book")

    # Load user_list and item_list
    user_list = pd.read_csv(os.path.join(ab, 'user_list.txt'), sep=' ')
    item_list = pd.read_csv(os.path.join(ab, 'item_list.txt'), sep=' ')

    def load_txt(path):
        users, items = [], []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                uid = int(parts[0])
                for iid in parts[1:]:
                    users.append(uid)
                    items.append(int(iid))
        return pd.DataFrame({'user_id': users, 'item_id': items})

    train_df = load_txt(os.path.join(ab, 'train.txt'))
    test_df = load_txt(os.path.join(ab, 'test.txt'))
    all_df = pd.concat([train_df, test_df])

    # 1. Missing values
    print("\n--- 1. Giá trị thiếu (Missing Values) ---")
    print(f"  user_list.txt null: {user_list.isnull().sum().sum()}")
    print(f"  item_list.txt null: {item_list.isnull().sum().sum()}")
    print(f"  train.txt null:     {train_df.isnull().sum().sum()}")
    print(f"  test.txt null:      {test_df.isnull().sum().sum()}")

    # 2. Duplicates
    print("\n--- 2. Bản ghi trùng lặp (Duplicates) ---")
    dup_train = train_df.duplicated().sum()
    dup_test = test_df.duplicated().sum()
    print(f"  train trùng lặp (user_id, item_id): {dup_train}")
    print(f"  test trùng lặp (user_id, item_id):  {dup_test}")

    # 3. Referential integrity
    print("\n--- 3. Tính toàn vẹn tham chiếu (Referential Integrity) ---")
    all_user_ids = set(all_df['user_id'].unique())
    mapped_user_ids = set(user_list['remap_id'].unique())
    orphan_users = all_user_ids - mapped_user_ids
    print(f"  User IDs trong interactions nhưng KHÔNG có trong user_list: {len(orphan_users)}")

    all_item_ids = set(all_df['item_id'].unique())
    mapped_item_ids = set(item_list['remap_id'].unique())
    orphan_items = all_item_ids - mapped_item_ids
    print(f"  Item IDs trong interactions nhưng KHÔNG có trong item_list: {len(orphan_items)}")

    # 4. Train/Test overlap
    print("\n--- 4. Rò rỉ dữ liệu Train/Test (Data Leakage) ---")
    train_pairs = set(zip(train_df['user_id'], train_df['item_id']))
    test_pairs = set(zip(test_df['user_id'], test_df['item_id']))
    overlap = train_pairs & test_pairs
    print(f"  Cặp (user, item) trùng giữa train và test: {len(overlap)}")
    print(f"  {'✅ Không rò rỉ' if len(overlap) == 0 else '⚠️ CÓ rò rỉ dữ liệu!'}")

    # 5. Cold-start check
    print("\n--- 5. Kiểm tra Cold-start ---")
    train_users = set(train_df['user_id'].unique())
    test_users = set(test_df['user_id'].unique())
    cold_users = test_users - train_users
    print(f"  Cold-start users (trong test nhưng không trong train): {len(cold_users)}")

    train_items = set(train_df['item_id'].unique())
    test_items = set(test_df['item_id'].unique())
    cold_items = test_items - train_items
    print(f"  Cold-start items (trong test nhưng không trong train): {len(cold_items)}")

    # 6. Preprocessed data integrity
    print("\n--- 6. Kiểm tra dữ liệu đã tiền xử lý ---")
    pp = os.path.join(ab, 'preprocessed')
    pp_train = pd.read_csv(os.path.join(pp, 'train.csv'))
    pp_test = pd.read_csv(os.path.join(pp, 'test.csv'))

    print(f"  train.csv rows: {len(pp_train)} (gốc: {len(train_df)}) {'✅' if len(pp_train) == len(train_df) else '❌'}")
    print(f"  test.csv rows:  {len(pp_test)} (gốc: {len(test_df)}) {'✅' if len(pp_test) == len(test_df) else '❌'}")
    print(f"  train.csv null: {pp_train.isnull().sum().sum()}")
    print(f"  test.csv null:  {pp_test.isnull().sum().sum()}")


if __name__ == "__main__":
    check_movielens()
    check_amazon()
    print("\n" + "=" * 70)
    print("✅ HOÀN THÀNH KIỂM TRA TÍNH TOÀN VẸN DỮ LIỆU")
    print("=" * 70)
