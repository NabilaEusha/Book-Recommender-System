import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load data
books = pd.read_csv('Books.csv')
ratings = pd.read_csv('Ratings.csv')

# Generate popular books
ratings_with_name = ratings.merge(books, on='ISBN')
num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)
avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_rating_df.rename(columns={'Book-Rating': 'avg_ratings'}, inplace=True)

popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
popular_df = popular_df[popular_df['num_ratings'] >= 250].sort_values('avg_ratings', ascending=False).head(50)
popular_df = popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_ratings']]

# Generate collaborative filtering
x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
poraku_users = x[x].index
filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(poraku_users)]

y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
famous_books = y[y].index
final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)

# Use float32 to save space
similarity_scores = cosine_similarity(pt).astype('float32')

# Keep only essential book columns
books_light = books[['Book-Title', 'Book-Author', 'Image-URL-M']].drop_duplicates('Book-Title')

# Save with compression
pickle.dump(popular_df, open('popular.pkl', 'wb'), protocol=4)
pickle.dump(pt, open('pt.pkl', 'wb'), protocol=4)
pickle.dump(books_light, open('books.pkl', 'wb'), protocol=4)
pickle.dump(similarity_scores, open('similarity_score.pkl', 'wb'), protocol=4)

print("✓ Optimized pickle files generated successfully!")