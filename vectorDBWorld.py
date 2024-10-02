import getpass
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Qdrant
import qdrant_client
import nest_asyncio
import json
import warnings
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Configurations
load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")
warnings.filterwarnings("ignore")
nest_asyncio.apply()
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")

# Grouped URLs by region
tourism_urls = {
    "europe-tour": [
        'https://www.planetware.com/tourist-attractions-/madrid-e-mad-mad.htm',
        'https://www.planetware.com/tourist-attractions-/barcelona-e-cat-bar.htm',
        'https://www.planetware.com/tourist-attractions-/milan-i-lo-m.htm',
        'https://www.planetware.com/tourist-attractions-/monaco-mc-mc-mon.htm',
        'https://www.trawell.in/blog/stunning-places-to-visit-in-berlin',
        'https://www.planetware.com/tourist-attractions-/prague-cz-pr-p.htm',
    ],
    "us-tour": [
        'https://www.planetware.com/tourist-attractions-/new-york-city-us-nyc.htm',
        'https://www.planetware.com/tourist-attractions-/los-angeles-us-ca-lam.htm',
        'https://www.planetware.com/tourist-attractions-/chicago-us-il-chic.htm',
        'https://www.planetware.com/tourist-attractions-/san-francisco-us-ca-sf.htm',
    ],
    "india-tour": [
        'https://www.neverendingfootsteps.com/three-days-new-delhi-itinerary',
        'https://www.travelandleisure.com/travel-guide/jaipur-india',
        'hhttps://www.nomadasaurus.com/first-timers-guide-goa-india',
        'https://thilo-hermann.medium.com/mumbai-in-a-nutshell-for-non-indians-sightseeing-44a2e7a0d434',
    ],
    "uk-tour": [
        'https://www.planetware.com/tourist-attractions-/edinburgh-sco-edn.htm',
        'https://www.planetware.com/tourist-attractions-/manchester-eng-man.htm',
        'https://www.planetware.com/tourist-attractions-/liverpool-eng-lvp.htm',
        'https://www.planetware.com/tourist-attractions-/oxford-eng-ox.htm',
    ]
}

# Gemini AI vector Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Gemini AI model
llm = ChatGoogleGenerativeAI(model="gemini-pro", gemini_api_key=gemini_api_key)

# Web Scrapping loader 
def webscrapper(urls: list):
    # load URLs
    loader = AsyncChromiumLoader(urls)
    docs = loader.load()

    # Apply BS4 transformer
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        # Extract content from given tags
        docs, tags_to_extract=["p", "h2", "span"]
    )
    
    # Perform Tokenization using Text Splitter
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=3000, 
        chunk_overlap=0
    )
    print('\n> Splitting documents into chunks')
    chunks = splitter.split_documents(docs_transformed)
    return chunks

# Create Qdrant Collection to store vector embeddings
# Modify the qdrant_collection function to upsert new documents
# Modify the qdrant_collection function to upsert new documents
def qdrant_collection(text_chunks, embedding_model, collection_name):
    print(f"> Connecting to QdrantDB for collection: {collection_name}")
    
    # Create a Qdrant Client
    client = qdrant_client.QdrantClient(
        qdrant_url, 
        api_key=qdrant_api_key
    )
    print(">\nQdrant connection established.")
    
    # Check if the collection already exists by listing collections
    existing_collections = client.get_collections().collections
    collection_exists = any(collection.name == collection_name for collection in existing_collections)

    if collection_exists:
        print(f"> Collection '{collection_name}' already exists. Adding new documents.")
    else:
        print(f"> Collection '{collection_name}' not found. Creating new collection.")
        
        # Create a new collection if it doesn't exist
        vectors_config = qdrant_client.http.models.VectorParams(
            size=768,  # Define fixed size of chunk to store
            distance=qdrant_client.http.models.Distance.COSINE
        )
        
        # Create collection (only once, avoid using recreate)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config
        )
    
    # Add new chunks to the existing collection using upsert
    qdrant = Qdrant.from_documents(
        text_chunks,
        embedding_model,
        url=qdrant_url,
        api_key=qdrant_api_key,
        prefer_grpc=True,
        collection_name=collection_name
    )
    print(f"> New chunks of text added to Qdrant DB for collection: {collection_name}")



# Loop through each region's URLs, scrape and store them in the respective collection
for collection_name, urls in tourism_urls.items():
    print(f"\nWEB SCRAPPING AND VECTOR EMBEDDING PROCESS FOR {collection_name.upper()} BEGINS")
    docs = webscrapper(urls)
    qdrant_collection(docs, embeddings, collection_name=collection_name)
