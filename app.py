from langchain_qdrant import Qdrant
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv 
import os 
import textwrap
import qdrant_client
import streamlit as st
import random 
from PIL import Image
from yarl import Query

# Configurations
load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")

# Define prompt
sys_prompt = """
    You are helpful tourism search engine. Give information about that places by given query. Always answer as helpful and as relevant
    as possible. While being informative. Keep answer length about 100-200 words.
    
    
    If you don't know the answer to a question, please don't share false information.    
"""


instruction = """CONTEXT:\n\n {context}\n\nQuery: {question}\n"""

# Dictionary mapping location keywords to collection names
location_to_collection_mapping = {
    'goa': 'india-tour',
    'mumbai': 'india-tour',
    'delhi': 'india-tour',
    'jaipur': 'india-tour',
    'india': 'india-tour',
    
    'paris': 'europe-tour',
    'london': 'europe-tour',
    'berlin': 'europe-tour',
    'madrid': 'europe-tour',
    'barcelona': 'europe-tour',
    'europe': 'europe-tour',
    
    'new york': 'us-tour',
    'california': 'us-tour',
    'san francisco': 'us-tour',
    'us': 'us-tour',
    'america': 'us-tour',
    
    'tokyo': 'asia-tour',
    'beijing': 'asia-tour',
    'japan': 'asia-tour',
    'china': 'asia-tour',
    'asia': 'asia-tour'
}


def get_vector_store(query):
    # Convert the query to lowercase for easier matching
    query_lower = query.lower()

    # Initialize collection_name with a default value
    collection_name = 'europe-tour'  # Default to 'europe-tour' if no match is found

    # Dynamic keyword mapping to collections
    for location_keyword, collection in location_to_collection_mapping.items():
        if location_keyword in query_lower:
            collection_name = collection
            break

    # Connect to QdrantDB Cloud
    client = qdrant_client.QdrantClient(
        qdrant_url,
        api_key=qdrant_api_key
    )
    
    # Define Embeddings 
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Vector store for Retrieval based on selected collection
    vector_store = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings
    )
    
    return vector_store





# Format the prompt 
def get_prompt(sys_prompt,instruction):
    prompt_template =  sys_prompt + instruction
    return prompt_template

# Custom output parser
def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


# Return generated text and source in an output
def process_llm_response(llm_response, query):
    # Get parsed answer
    text = wrap_text_preserve_newlines(llm_response['result'])
    
    # Uncouple metadata and return it
    sources = []
    relevant_content = []

    for source in llm_response["source_documents"]:
        # Check if 'source' exists in the metadata
        if 'source' in source.metadata:
            sources.append(source.metadata['source'])
        
        # Check if the retrieved document content is relevant to the query
        if any(keyword in source.page_content.lower() for keyword in query.lower().split()):
            relevant_content.append(source.page_content)

    # If no relevant content is found, return a fallback response
    if not relevant_content:
        relevant_content.append("No relevant content found for the query.")

    # Concatenate the relevant content to pass it to the LLM
    context = "\n\n".join(relevant_content)

    # Return the text, context, and deduplicated list of sources (URLs)
    return text, list(set(sources))




if __name__ == '__main__':
    # Set Streamlit UI
    st.set_page_config(page_title="AI Tour Assistant")
   
    st.markdown("# AI Travel Assistant")
   
    image = Image.open('europe_banner.jpg')
    st.image(image, caption='by Karan Shingde', use_column_width=True)
   
    st.header("Tell us about your travel destination?")
   
    # Create text box for user query
    user_query = st.text_input("What place would you love to explore?")
    
    if user_query:  # Only proceed if there's a user query
        # Pass the user's query to get_vector_store
        vector_store = get_vector_store(user_query)  # Pass the query to the function
    
        # Using Gemini-Pro
        llm = ChatGoogleGenerativeAI(model="gemini-pro", gemini_api_key=gemini_api_key,
                                     temperature=0.3,
                                     max_tokens=1024,
                                     convert_system_message_to_human=True)
    
        # Generate Prompt Template
        prompt_template = get_prompt(instruction, sys_prompt)
        QA_prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
       
        # Create Retrieval Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,  # Get source
            chain_type_kwargs={"prompt": QA_prompt}
        )
        
        # Generate response
        llm_res = qa_chain.invoke(user_query)  # Generate response with the user's query
        
        # Process the response and context
        response, sources = process_llm_response(llm_res, user_query)  # Process the response
        
        st.markdown("### Based on your search:")
        st.write(f"{response}")
        st.markdown("##### Sources: ")
        for source in sources:  # Display source URLs
            st.markdown(f"[{source}]({source})", unsafe_allow_html=True)


