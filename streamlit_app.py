import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Pinecone, ServerlessSpec

# Replace with your actual Pinecone API key
api_key = "pcsk_2zkszn_UbXuT2K7parwLi3U2QEKmsw2uBr43kqgrFig4nEuC3b81CFRbeXdm98NDbc1GkR"

# Load the IMDB dataset from GitHub
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Rahulv024/semantic-search-app/main/imdb_top_1000.csv"
    st.write(f"Fetching dataset from: {url}")
    try:
        df = pd.read_csv(url)
        st.write("Dataset loaded successfully!")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()


df = load_data()

# Convert necessary columns to string and handle missing values
df['Series_Title'] = df['Series_Title'].astype(str).fillna("")
df['Genre'] = df['Genre'].astype(str).fillna("")
df['Overview'] = df['Overview'].astype(str).fillna("")

# Create the text field for embedding
df['text'] = df['Series_Title'] + " " + df['Genre'] + " " + df['Overview']

# Load Sentence Transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Generate embeddings
df['embedding'] = df['text'].apply(lambda x: model.encode(x).tolist())

# Initialize Pinecone
pc = Pinecone(api_key=api_key)

# Define index name
index_name = "imdb-movie-search"

# Check if index already exists, if not create one
if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,  # Must match the embedding size of your model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    st.write(f"Index '{index_name}' created successfully.")
else:
    st.write(f"Index '{index_name}' already exists.")

# Connect to Pinecone index
index = pc.Index(index_name)
st.write("Connected to Pinecone index.")

# Upload embeddings to Pinecone
for i, row in df.iterrows():
    index.upsert([(str(i), row['embedding'], {'title': row['Series_Title'], 'Genre': row['Genre'], 'overview': row['Overview']})])

st.write("Data uploaded to Pinecone.")

# Function to Search Movies
def search_movies(query, top_k=5):
    # Generate query embedding
    query_embedding = model.encode(query).tolist()

    # Search in Pinecone
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    # Extract and display results
    movie_results = []
    for match in results['matches']:
        movie_results.append({
            "Title": match['metadata']['title'],
            "Overview": match['metadata']['overview'],
            "Genre": match['metadata']['Genre']
        })

    return pd.DataFrame(movie_results)

# Streamlit UI
st.title("🎬 IMDB Movie Recommendation System")

# User input for search query
query = st.text_input("Describe the type of movie you want to watch:")

if query:
    search_results = search_movies(query)

    # Display results in Streamlit
    st.subheader("🎥 Recommended Movies:")
    st.dataframe(search_results)
