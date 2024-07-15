# import pandas as pd
# from elasticsearch import Elasticsearch, helpers
# import tensorflow as tf
# import tensorflow_recommenders as tfrs
# from tensorflow.keras.layers import StringLookup, Embedding, Dense
# import streamlit as st

# # Load and preprocess the CSV file
# df = pd.read_csv('products-all-data.csv')
# df = df[['products_id', 'products_name', 'category', 'sub_category', 'products_mrp', 'rating', 'rating_count', 'review_count', 'm_img']]
# df['products_id'] = df['products_id'].astype(str)
# df['m_img'] = 'https://cdn.igp.com/f_auto,q_auto,t_pnopt3prodlp/products/' + df['m_img']

# # Initialize Elasticsearch
# es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme':'http'}])

# # Function to index products into Elasticsearch
# def index_products(df):
#     actions = [
#         {
#             "_index": "products",
#             "_id": row['products_id'],
#             "_source": {
#                 "products_id": row['products_id'],
#                 "products_name": row['products_name'],
#                 "category": row['category'],
#                 "sub_category": row['sub_category'],
#                 "products_mrp": row['products_mrp'],
#                 "rating": row['rating'],
#                 "rating_count": row['rating_count'],
#                 "review_count": row['review_count'],
#                 "m_img": row['m_img']
#             }
#         }
#         for _, row in df.iterrows()
#     ]
#     helpers.bulk(es, actions)

# # Index the data
# index_products(df)

# # Define the ProductModel class
# class ProductModel(tfrs.Model):
#     def __init__(self, df):
#         super().__init__()
#         embedding_dimension = 32

#         self.product_model = tf.keras.Sequential([
#             StringLookup(vocabulary=df['products_id'].unique(), mask_token=None),
#             Embedding(len(df['products_id'].unique()) + 1, embedding_dimension)
#         ])
#         self.rating_model = tf.keras.Sequential([
#             Dense(256, activation="relu", kernel_regularizer=l2(0.01)),
#             Dense(64, activation="relu", kernel_regularizer=l2(0.01)),
#             Dense(1)
#         ])
#         self.task = tfrs.tasks.Ranking(
#             loss=tf.keras.losses.MeanSquaredError(),
#             metrics=[tf.keras.metrics.RootMeanSquaredError()]
#         )

#     def call(self, features):
#         product_embeddings = self.product_model(features["products_id"])
#         rating_predictions = self.rating_model(tf.concat([product_embeddings], axis=1))
#         return rating_predictions

#     def compute_loss(self, features, training=False):
#         labels = features["rating"]
#         predictions = self(features)
#         return self.task(labels=labels, predictions=predictions)

# # Load the trained model
# model = tf.keras.models.load_model('product_recommender_model', custom_objects={'ProductModel': ProductModel, 'StringLookup': StringLookup})

# # Define the function to get similar products by subcategory
# def get_similar_products_by_subcategory(subcategory, num_recommendations=5):
#     search_body = {
#         "query": {
#             "match": {
#                 "sub_category": subcategory
#             }
#         },
#         "size": 50
#     }
#     res = es.search(index="products", body=search_body)
#     if res['hits']['total']['value'] == 0:
#         return []
#     candidates = []
#     for hit in res['hits']['hits']:
#         candidate = {
#             "products_name": hit['_source']['products_name'],
#             "category": hit['_source']['category'],
#             "sub_category": hit['_source']['sub_category'],
#             "products_mrp": hit['_source']['products_mrp'],
#             "m_img": hit['_source'].get('m_img', None)
#         }
#         candidates.append(candidate)
#     sorted_candidates = sorted(candidates, key=lambda x: x["products_mrp"])
#     return sorted_candidates[:num_recommendations]

# # Streamlit app
# st.set_page_config(page_title="Product Recommendation System", layout="wide")
# st.title("Product Recommendation System")

# # Custom CSS for styling
# st.markdown("""
#     <style>
#     .reportview-container {
#         background: #f0f2f6;
#         padding: 20px;
#     }
#     .stButton>button {
#         background-color: #4CAF50;
#         color: white;
#         border-radius: 8px;
#         border: none;
#         padding: 10px 24px;
#         font-size: 16px;
#     }
#     .stTextInput>div>div>input {
#         border-radius: 8px;
#         border: 2px solid #ccc;
#         padding: 10px;
#         font-size: 16px;
#     }
#     .recommendation-card {
#         background: white;
#         padding: 20px;
#         border-radius: 8px;
#         box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
#         margin-bottom: 20px;
#         display: flex;
#         flex-direction: column;
#         align-items: center;
#     }
#     .recommendation-card img {
#         border-radius: 8px;
#         width: 100px;
#         height: 100px;
#     }
#     .recommendation-card .details {
#         text-align: center;
#         margin-top: 10px;
#     }
#     .recommendation-card .details .price {
#         color: #e74c3c;
#         font-size: 18px;
#         font-weight: bold;
#     }
#     </style>
# """, unsafe_allow_html=True)

# subcategory = st.text_input("Enter a subcategory to get recommendations:", "home decor")
# num_recommendations = st.text_input("Enter the number of recommendations required:", "5")

# if st.button("Get Recommendations"):
#     try:
#         num_recommendations = int(num_recommendations)
#         recommendations = get_similar_products_by_subcategory(subcategory, num_recommendations)
#         if recommendations:
#             st.write("Recommendations:")
#             for rec in recommendations:
#                 cols = st.columns(5)
#                 with cols[0]:
#                     st.image(rec['m_img'], width=100)
#                 with cols[1]:
#                     st.write(f"**{rec['products_name']}**")
#                 with cols[2]:
#                     st.write(f"**{rec['category']}**")
#                 with cols[3]:
#                     st.write(f"**{rec['sub_category']}**")
#                 with cols[4]:
#                     st.write(f"**₹{rec['products_mrp']}**")
#                 st.write("---")
#         else:
#             st.write("No recommendations found.")
#     except ValueError:
#         st.write("Please enter a valid number for recommendations.")

import pandas as pd
from elasticsearch import Elasticsearch, helpers
import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow.keras.layers import StringLookup, Embedding, Dense
import streamlit as st

# Load and preprocess the CSV file
df = pd.read_csv('products-all-data.csv')
df = df[['products_id', 'products_name', 'category', 'sub_category', 'products_mrp', 'rating', 'rating_count', 'review_count', 'm_img']]
df['products_id'] = df['products_id'].astype(str)
df['m_img'] = 'https://cdn.igp.com/f_auto,q_auto,t_pnopt3prodlp/products/' + df['m_img']

# Initialize Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme':'http'}])

# Function to index products into Elasticsearch
def index_products(df):
    actions = [
        {
            "_index": "products",
            "_id": row['products_id'],
            "_source": {
                "products_id": row['products_id'],
                "products_name": row['products_name'],
                "category": row['category'],
                "sub_category": row['sub_category'],
                "products_mrp": row['products_mrp'],
                "rating": row['rating'],
                "rating_count": row['rating_count'],
                "review_count": row['review_count'],
                "m_img": row['m_img']
            }
        }
        for _, row in df.iterrows()
    ]
    helpers.bulk(es, actions)

# Index the data
index_products(df)

# Define the ProductModel class
class ProductModel(tfrs.Model):
    def __init__(self, df):
        super().__init__()
        embedding_dimension = 32

        self.product_model = tf.keras.Sequential([
            StringLookup(vocabulary=df['products_id'].unique(), mask_token=None),
            Embedding(len(df['products_id'].unique()) + 1, embedding_dimension)
        ])
        self.rating_model = tf.keras.Sequential([
            Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            Dense(1)
        ])
        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, features):
        product_embeddings = self.product_model(features["products_id"])
        rating_predictions = self.rating_model(tf.concat([product_embeddings], axis=1))
        return rating_predictions

    def compute_loss(self, features, training=False):
        labels = features["rating"]
        predictions = self(features)
        return self.task(labels=labels, predictions=predictions)

# Load the trained model
model = tf.keras.models.load_model('product_recommender_model', custom_objects={'ProductModel': ProductModel, 'StringLookup': StringLookup})

# Define the function to get similar products by subcategory
def get_similar_products_by_subcategory(subcategory, num_recommendations=5):
    search_body = {
        "query": {
            "match": {
                "sub_category": subcategory
            }
        },
        "size": 50
    }
    res = es.search(index="products", body=search_body)
    if res['hits']['total']['value'] == 0:
        return []
    candidates = []
    for hit in res['hits']['hits']:
        candidate = {
            "products_name": hit['_source']['products_name'],
            "category": hit['_source']['category'],
            "sub_category": hit['_source']['sub_category'],
            "products_mrp": hit['_source']['products_mrp'],
            "m_img": hit['_source'].get('m_img', None)
        }
        candidates.append(candidate)
    sorted_candidates = sorted(candidates, key=lambda x: x["products_mrp"])
    return sorted_candidates[:num_recommendations]
# Streamlit app
st.set_page_config(page_title="ShopSmart Recommender", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-image: url('https://images.unsplash.com/photo-1607082348824-0a96f2a4b9da?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.85);
    }
    .main-container {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 2px solid #ccc;
        padding: 10px;
        font-size: 16px;
    }
    .product-card {
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        padding: 15px;
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
    }
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .product-card img {
        width: 100%;
        height: 250px;
        object-fit: contain;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .product-name {
        font-size: 16px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 5px;
    }
    .product-category, .product-subcategory {
        font-size: 14px;
        color: #7f8c8d;
        margin-bottom: 5px;
    }
    .product-price {
        font-size: 18px;
        font-weight: bold;
        color: #e74c3c;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 30px;
        font-size: 36px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🛍️ ShopSmart Recommender</h1>", unsafe_allow_html=True)

# Main container
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

# Inputs and button in a centered container
with st.container():
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        subcategory = st.text_input("Enter a subcategory:", "home decor")
    with col2:
        num_recommendations = st.text_input("Number of recommendations:", "8")
    with col3:
        find_products_button = st.button("🔍 Find Products")

if find_products_button:
    try:
        num_recommendations = int(num_recommendations)
        recommendations = get_similar_products_by_subcategory(subcategory, num_recommendations)
        if recommendations:
            st.write(f"Showing results for: **{subcategory}** ({len(recommendations)} products)")
            st.write("Top Picks for You:")
            
            # Display recommendations in a grid with 4 cards per row
            for i in range(0, len(recommendations), 4):
                cols = st.columns(4)
                for j in range(4):
                    if i + j < len(recommendations):
                        rec = recommendations[i + j]
                        with cols[j]:
                            # Modify the image URL to request a higher quality image
                            high_quality_img_url = rec['m_img'].replace('q_auto', 'q_80')
                            st.markdown(
                                f"""
                                <div class='product-card'>
                                    <img src="{high_quality_img_url}" alt="{rec['products_name']}">
                                    <p class='product-name'>{rec['products_name']}</p>
                                    <p class='product-category'>{rec['category']}</p>
                                    <p class='product-subcategory'>{rec['sub_category']}</p>
                                    <p class='product-price'>₹{rec['products_mrp']}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
        else:
            st.write("No recommendations found. Try a different subcategory!")
    except ValueError:
        st.write("Please enter a valid number for recommendations.")

st.markdown("</div>", unsafe_allow_html=True)