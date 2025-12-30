from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__, template_folder='../templates')

# Get the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load pickle files
popular_df = pickle.load(open(os.path.join(BASE_DIR, 'popular.pkl'), 'rb'))
pt = pickle.load(open(os.path.join(BASE_DIR, 'pt.pkl'), 'rb'))
books = pickle.load(open(os.path.join(BASE_DIR, 'books.pkl'), 'rb'))
similarity_scores = pickle.load(open(os.path.join(BASE_DIR, 'similarity_score.pkl'), 'rb'))

@app.route('/')
def index():
    return render_template('index.html',
                           book_name=list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_ratings'].values))

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books', methods=['post'])
def recommend():
    user_input = request.form.get('user_input')

    if user_input not in pt.index:
        return render_template('recommend.html', data=None,
                             error="Book not found. Try an exact title from the dataset.")

    index = np.where(pt.index == user_input)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])),
                          key=lambda x: x[1], reverse=True)[1:9]

    data = []
    for i in similar_items:
        temp_df = books[books['Book-Title'] == pt.index[i[0]]].drop_duplicates('Book-Title')
        item = [
            temp_df['Book-Title'].values[0],
            temp_df['Book-Author'].values[0],
            temp_df['Image-URL-M'].values[0]
        ]
        data.append(item)

    return render_template('recommend.html', data=data)

@app.route('/contact')
def contact():
    return render_template('contact.html')

# This is important for Vercel
app = app