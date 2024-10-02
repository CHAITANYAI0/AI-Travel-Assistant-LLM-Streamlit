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
from PIL import Image
from fpdf import FPDF  # Import for generating PDF files
import time

# Configurations
load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")

# Define prompt
sys_prompt = """
    You are a helpful tourism search engine. Provide information about places based on the given query. Always answer as helpfully and as relevant
    as possible while being informative. Keep answer length between 100-200 words.
    
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

# Function to select the vector store based on the query
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
def get_prompt(sys_prompt, instruction):
    prompt_template = sys_prompt + instruction
    return prompt_template

# Custom output parser to wrap text properly
def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

# Function to process the LLM response and return text and sources
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

# Updated function to get the best places to visit using RetrievalQA with LLM and retriever
def get_best_places_to_visit(destination, qa_chain):
    query = f"Provide a list of the best places to visit in {destination}. Briefly describe each place."
    
    # Get response from RetrievalQA
    result = qa_chain({"query": query})  # Use __call__() to handle multiple outputs
    
    # Process the LLM response using the process_llm_response function
    text, sources = process_llm_response(result, query)
    
    # Return the generated text (best places to visit) and the sources
    return text.split('\n'), sources


def generate_itinerary_with_places(destination, days, llm):
    visited_places = set()  # Track places already visited
    itinerary = []
    
    # For each day, generate a query asking for specific places to visit
    for day in range(1, days + 1):
        # Modify the query to ask for unique places with recommended time slots
        detailed_query = f"""
        You are a travel assistant. Provide a detailed itinerary for day {day} of a visit to {destination}.
        Suggest 2 unique places for the morning, 2 for the afternoon, and 2 for the evening.
        Include visiting hours or recommended time slots for each place. Ensure no places are repeated from previous days.
        """
        
        # Get response from LLM
        response = llm.predict(detailed_query)  # Using .predict to get a string response

        # Check if the response is in string format
        if isinstance(response, str):
            # Split the response into lines and prepare to organize into morning, afternoon, evening
            lines = response.strip().split('\n')
            morning_places, afternoon_places, evening_places = [], [], []

            # Process each line to avoid repetition and properly structure the output
            current_time_slot = None
            for line in lines:
                line_content = line.strip().lower()
                
                # Determine the time slot (morning, afternoon, evening) based on the line
                if 'morning' in line_content:
                    current_time_slot = 'morning'
                elif 'afternoon' in line_content:
                    current_time_slot = 'afternoon'
                elif 'evening' in line_content:
                    current_time_slot = 'evening'

                # Add the place to the respective time slot if it's not already visited
                if line_content not in visited_places and line_content:
                    visited_places.add(line_content)
                    if current_time_slot == 'morning':
                        morning_places.append(line.strip())
                    elif current_time_slot == 'afternoon':
                        afternoon_places.append(line.strip())
                    elif current_time_slot == 'evening':
                        evening_places.append(line.strip())

            # Fallback for time slots with no valid places
            if not morning_places:
                morning_places = ["**Morning (9:00 AM - 12:00 PM):** No activity found."]
            if not afternoon_places:
                afternoon_places = ["**Afternoon (1:00 PM - 4:00 PM):** No activity found."]
            if not evening_places:
                evening_places = ["**Evening (6:00 PM - 9:00 PM):** No activity found."]

            # Append the structured itinerary for the day with proper formatting
            itinerary.append(f"**Day {day}:**")
            itinerary.append("**Morning:**")
            itinerary.extend([f"- {place}" for place in morning_places])
            itinerary.append("\n**Afternoon:**")
            itinerary.extend([f"- {place}" for place in afternoon_places])
            itinerary.append("\n**Evening:**")
            itinerary.extend([f"- {place}" for place in evening_places])
            itinerary.append("\n")  # Add an extra line for spacing
        else:
            itinerary.append(f"**Day {day}:** No valid response from LLM.")
    
    return itinerary






# Main Streamlit app logic
if __name__ == '__main__':
    st.set_page_config(page_title="AI Tour Assistant")
   
    st.markdown("# AI Travel Assistant")
   
    image = Image.open('AI_travel.jpg')
    st.image(image, caption='by Chaitanya Inamdar', use_column_width=True)
   
    st.header("Tell us about your travel destination?")
   
    # Initialize session state to manage visibility of forms
    if 'show_best_places_form' not in st.session_state:
        st.session_state.show_best_places_form = False
    if 'show_itinerary_form' not in st.session_state:
        st.session_state.show_itinerary_form = False

    # Two buttons to open the respective forms
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Get Best Places to Visit"):
            st.session_state.show_best_places_form = True
            st.session_state.show_itinerary_form = False  # Hide the other form

    with col2:
        if st.button("Generate Itinerary"):
            st.session_state.show_itinerary_form = True
            st.session_state.show_best_places_form = False  # Hide the other form

    # Show "Best Places to Visit" form if the button is clicked
    if st.session_state.show_best_places_form:
        st.subheader("Find the Best Places to Visit")
        # Create form for user input
        with st.form(key="best_places_form"):
            destination = st.text_input("What place would you love to explore?")
            interests = st.multiselect(
                "Select your interests:",
                options=['History', 'Culture', 'Adventure', 'Nature', 'Food', 'Shopping'],
                help="Pick the categories that interest you."
            )
            submit_best_places = st.form_submit_button(label="Submit Best Places")
        
        if submit_best_places and destination:
            # Assuming you have the LLM initialized
            vector_store = get_vector_store(destination)  # Pass the query to the function

            llm = ChatGoogleGenerativeAI(model="gemini-pro", gemini_api_key="your_gemini_api_key", 
                                         temperature=0.3, max_tokens=1024,
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

            # Incorporate interests into the query if any interests were selected
            interests_str = ', '.join(interests) if interests else "general"
            user_query = f"Provide a list of the best places to visit in {destination} based on interests in {interests_str}. Briefly describe each place."

            # Create a progress bar with airplane emoji
            progress = st.progress(0)

            # Start loading bar as query is prepared
            with st.spinner("✈️ Preparing your results..."):
                progress.progress(20)  # Progress after the query is prepared

                # Generate response
                llm_res = qa_chain.invoke(user_query)  # Generate response with the user's query
                progress.progress(50)  # Halfway as query is processed

                # Process the response and context
                response, sources = process_llm_response(llm_res, user_query)  # Process the response
                progress.progress(80)  # Progress after response is processed

            # Display the results in Streamlit
            st.markdown("### Based on your search for the best places to visit:")
            st.write(f"{response}")
            progress.progress(100)  # Full progress when results are displayed

    # Show "Generate Itinerary" form if the button is clicked
    if st.session_state.show_itinerary_form:
        st.subheader("Create Your Itinerary")
        # Create form for user input
        with st.form(key="itinerary_form"):
            destination = st.text_input("What place are you visiting?")
            days_to_stay = st.number_input("How many days will you stay?", min_value=1, step=1)
            submit_itinerary = st.form_submit_button(label="Submit Itinerary")
        
        if submit_itinerary and destination and days_to_stay:
            # Assuming you have the LLM initialized
            vector_store = get_vector_store(destination)  # Pass the query to the function

            llm = ChatGoogleGenerativeAI(model="gemini-pro", gemini_api_key="your_gemini_api_key", 
                                         temperature=0.3, max_tokens=1024,
                                         convert_system_message_to_human=True)

            # Create a progress bar with airplane emoji
            progress = st.progress(0)

            with st.spinner("✈️ Generating your itinerary..."):
                progress.progress(20)  # Initial progress when starting itinerary generation

                # Use the provided `generate_itinerary_with_places` function
                itinerary = generate_itinerary_with_places(destination, days_to_stay, llm)
                progress.progress(80)  # Progress after itinerary is generated

            # Display the generated itinerary
            st.markdown("### Your Itinerary:")
            for day_plan in itinerary:
                st.markdown(day_plan)
            progress.progress(100)  # Full progress when results are displayed












