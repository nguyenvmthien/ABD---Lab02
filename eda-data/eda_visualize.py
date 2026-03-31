import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eda_plots")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

# ============================================================
# MOVIELENS 1M
# ============================================================
def plot_movielens():
    ml = os.path.join(BASE, "movielens")
    ratings = pd.read_csv(os.path.join(ml, 'ratings.dat'), sep='::', engine='python',
                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'], encoding='ISO-8859-1')
    movies = pd.read_csv(os.path.join(ml, 'movies.dat'), sep='::', engine='python',
                         names=['MovieID', 'Title', 'Genres'], encoding='ISO-8859-1')
    users = pd.read_csv(os.path.join(ml, 'users.dat'), sep='::', engine='python',
                        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], encoding='ISO-8859-1')

    # 1. Rating Distribution
    fig, ax = plt.subplots()
    counts = ratings['Rating'].value_counts().sort_index()
    bars = ax.bar(counts.index, counts.values, color=['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db'], edgecolor='black')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Số lượng')
    ax.set_title('MovieLens 1M - Phân phối Rating')
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 5000, f'{v:,}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'ml_rating_distribution.png'), dpi=150)
    plt.close()
    print("  Saved: ml_rating_distribution.png")

    # 2. Ratings per User (histogram)
    fig, ax = plt.subplots()
    ratings_per_user = ratings.groupby('UserID').size()
    ax.hist(ratings_per_user, bins=50, color='#3498db', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Số lượng ratings')
    ax.set_ylabel('Số lượng người dùng')
    ax.set_title('MovieLens 1M - Phân phối số ratings mỗi người dùng')
    ax.axvline(ratings_per_user.mean(), color='red', linestyle='--', label=f'Trung bình: {ratings_per_user.mean():.0f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'ml_ratings_per_user.png'), dpi=150)
    plt.close()
    print("  Saved: ml_ratings_per_user.png")

    # 3. Ratings per Movie (histogram)
    fig, ax = plt.subplots()
    ratings_per_movie = ratings.groupby('MovieID').size()
    ax.hist(ratings_per_movie, bins=50, color='#2ecc71', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Số lượng ratings')
    ax.set_ylabel('Số lượng phim')
    ax.set_title('MovieLens 1M - Phân phối số ratings mỗi phim')
    ax.axvline(ratings_per_movie.mean(), color='red', linestyle='--', label=f'Trung bình: {ratings_per_movie.mean():.0f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'ml_ratings_per_movie.png'), dpi=150)
    plt.close()
    print("  Saved: ml_ratings_per_movie.png")

    # 4. Top 10 most rated movies
    fig, ax = plt.subplots(figsize=(12, 6))
    top10 = ratings.groupby('MovieID').size().sort_values(ascending=False).head(10)
    titles = []
    for mid in top10.index:
        t = movies[movies['MovieID'] == mid]['Title'].values
        title = t[0] if len(t) > 0 else str(mid)
        if len(title) > 35:
            title = title[:32] + '...'
        titles.append(title)
    bars = ax.barh(range(len(titles)), top10.values, color='#9b59b6', edgecolor='black')
    ax.set_yticks(range(len(titles)))
    ax.set_yticklabels(titles, fontsize=10)
    ax.set_xlabel('Số lượng ratings')
    ax.set_title('MovieLens 1M - Top 10 phim được đánh giá nhiều nhất')
    ax.invert_yaxis()
    for bar, v in zip(bars, top10.values):
        ax.text(v + 20, bar.get_y() + bar.get_height()/2, f'{v:,}', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'ml_top10_movies.png'), dpi=150)
    plt.close()
    print("  Saved: ml_top10_movies.png")

    # 5. Gender distribution
    fig, ax = plt.subplots()
    gender_counts = users['Gender'].value_counts()
    ax.pie(gender_counts.values, labels=['Nam (M)', 'Nữ (F)'] if gender_counts.index[0] == 'M' else ['Nữ (F)', 'Nam (M)'],
           autopct='%1.1f%%', colors=['#3498db', '#e74c3c'], startangle=90)
    ax.set_title('MovieLens 1M - Phân phối giới tính')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'ml_gender_distribution.png'), dpi=150)
    plt.close()
    print("  Saved: ml_gender_distribution.png")

    # 6. Age distribution
    fig, ax = plt.subplots()
    age_map = {1: '<18', 18: '18-24', 25: '25-34', 35: '35-44', 45: '45-49', 50: '50-55', 56: '56+'}
    users['AgeGroup'] = users['Age'].map(age_map)
    age_order = ['<18', '18-24', '25-34', '35-44', '45-49', '50-55', '56+']
    age_counts = users['AgeGroup'].value_counts().reindex(age_order)
    bars = ax.bar(age_counts.index, age_counts.values, color='#e67e22', edgecolor='black')
    ax.set_xlabel('Nhóm tuổi')
    ax.set_ylabel('Số lượng')
    ax.set_title('MovieLens 1M - Phân phối nhóm tuổi')
    for bar, v in zip(bars, age_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 20, f'{v:,}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'ml_age_distribution.png'), dpi=150)
    plt.close()
    print("  Saved: ml_age_distribution.png")


# ============================================================
# AMAZON-BOOK
# ============================================================
def plot_amazon():
    ab = os.path.join(BASE, "amazon-book")

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

    # 1. Interactions per User
    fig, ax = plt.subplots()
    user_counts = all_df.groupby('user_id').size()
    ax.hist(user_counts, bins=50, color='#3498db', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Số lượng tương tác')
    ax.set_ylabel('Số lượng người dùng')
    ax.set_title('Amazon-Book - Phân phối số tương tác mỗi người dùng')
    ax.axvline(user_counts.mean(), color='red', linestyle='--', label=f'Trung bình: {user_counts.mean():.0f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'ab_interactions_per_user.png'), dpi=150)
    plt.close()
    print("  Saved: ab_interactions_per_user.png")

    # 2. Interactions per Item
    fig, ax = plt.subplots()
    item_counts = all_df.groupby('item_id').size()
    ax.hist(item_counts, bins=50, color='#2ecc71', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Số lượng tương tác')
    ax.set_ylabel('Số lượng sản phẩm')
    ax.set_title('Amazon-Book - Phân phối số tương tác mỗi sản phẩm')
    ax.axvline(item_counts.mean(), color='red', linestyle='--', label=f'Trung bình: {item_counts.mean():.0f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'ab_interactions_per_item.png'), dpi=150)
    plt.close()
    print("  Saved: ab_interactions_per_item.png")

    # 3. Train vs Test split (Pie)
    fig, ax = plt.subplots()
    ax.pie([len(train_df), len(test_df)],
           labels=[f'Train ({len(train_df):,})', f'Test ({len(test_df):,})'],
           autopct='%1.1f%%', colors=['#3498db', '#e74c3c'], startangle=90)
    ax.set_title('Amazon-Book - Tỷ lệ Train/Test')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'ab_train_test_split.png'), dpi=150)
    plt.close()
    print("  Saved: ab_train_test_split.png")

    # 4. Comparison: MovieLens vs Amazon-Book
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ml = os.path.join(BASE, "movielens")
    ml_ratings = pd.read_csv(os.path.join(ml, 'ratings.dat'), sep='::', engine='python',
                             names=['UserID', 'MovieID', 'Rating', 'Timestamp'], encoding='ISO-8859-1')
    ml_users = ml_ratings['UserID'].nunique()
    ml_items = ml_ratings['MovieID'].nunique()
    ml_total = len(ml_ratings)
    ab_users = all_df['user_id'].nunique()
    ab_items = all_df['item_id'].nunique()
    ab_total = len(all_df)

    # Subplot 1: User/Item/Interaction counts
    x = np.arange(3)
    w = 0.35
    bars1 = axes[0].bar(x - w/2, [ml_users, ml_items, ml_total], w, label='MovieLens 1M', color='#3498db', edgecolor='black')
    bars2 = axes[0].bar(x + w/2, [ab_users, ab_items, ab_total], w, label='Amazon-Book', color='#e74c3c', edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Users', 'Items', 'Interactions'])
    axes[0].set_title('So sánh qui mô')
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Số lượng (log scale)')

    # Subplot 2: Sparsity comparison
    ml_sparsity = 1 - ml_total / (ml_users * ml_items)
    ab_sparsity = 1 - ab_total / (ab_users * ab_items)
    bars = axes[1].bar(['MovieLens 1M', 'Amazon-Book'], [ml_sparsity * 100, ab_sparsity * 100],
                       color=['#3498db', '#e74c3c'], edgecolor='black')
    axes[1].set_ylabel('Sparsity (%)')
    axes[1].set_title('So sánh độ thưa thớt')
    axes[1].set_ylim(90, 100)
    for bar, v in zip(bars, [ml_sparsity * 100, ab_sparsity * 100]):
        axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.1, f'{v:.2f}%', ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'comparison_ml_vs_ab.png'), dpi=150)
    plt.close()
    print("  Saved: comparison_ml_vs_ab.png")


if __name__ == "__main__":
    print("Generating MovieLens 1M plots...")
    plot_movielens()
    print("\nGenerating Amazon-Book plots...")
    plot_amazon()
    print(f"\n✅ All plots saved to: {OUT_DIR}")
