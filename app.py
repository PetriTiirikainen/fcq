import pandas as pd
import os
import streamlit as st
from sklearn.neighbors import NearestNeighbors
import numpy as np
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession, ResponseValidationError
from constants import TRAIT_LIST, OPTIONS, TRAIT_PAIR_LIST
from filters import universes, genres, genders, filter_by_gender, filter_by_universe, filter_by_genre
from chatbot import chatbot_init, get_chat_response
from dotenv import load_dotenv

# Set the path to the Google credentials file
credentials_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file

# Set the path to the Google credentials file
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ".\\keys\\wbs-final-project-412007-0e822eb70e15.json"

# Load the data
matrix_df = pd.read_csv('matrix.csv').set_index('char_id')
characters_df = pd.read_csv('characters.csv')
psych_stats1_df = pd.read_csv('psych_stats1.csv')
psych_stats2_df = pd.read_csv('psych_stats2.csv')
uni_matrix_df = pd.read_csv('uni_matrix.csv').set_index('uni_name')

# Merge the dataframes
psych_stats_df = pd.concat([psych_stats1_df, psych_stats2_df], axis=0)

# Adding the title with pink color directly after your CSS block
st.markdown('<h1 style="color: #FF0000;">Fictio Chat Quest</h1>', unsafe_allow_html=True)
st.write("")
st.caption("This is a fun app to find your character and universe based on your personality traits. You can also chat with the character you get! The more questions you decide to answer, the more accurate the result will be, okay? Let's get started!") 
st.write("")

# Columns for filters
col1, col2, col3, col4 = st.columns(4)

with col1:
    selected_option = st.selectbox('How many questions?:', list(OPTIONS.keys()), index=0, key="lod")

with col2:
    selected_universe = st.selectbox('Which universe from?:', universes, index=0, key="universe")

with col3:
    selected_genre = st.selectbox('Which genre?:', genres, index=0, key="genre")

with col4:
    selected_gender = st.selectbox('Which gender?:', genders, index=0, key="gender")

# Get the corresponding value of N
N = OPTIONS[selected_option]

# Select only as many traits as the user wanted
selected_features = TRAIT_LIST[:N]

characters_df, matrix_df = filter_by_gender(characters_df, matrix_df, selected_gender)
characters_df, matrix_df = filter_by_universe(selected_universe, characters_df, matrix_df)
characters_df, matrix_df = filter_by_genre(selected_genre, characters_df, matrix_df)

# Create a NearestNeighbors model
def create_nn_model(matrix_df, selected_features):
    nn_model = NearestNeighbors(n_neighbors=1)
    X = matrix_df[selected_features].fillna(0)
    nn_model.fit(X)
    return nn_model

nn_model = create_nn_model(matrix_df, selected_features)
st.write("")
st.markdown('<div style="text-align: center">On a scale of 1-100, where do you fall between these words??:</div>', unsafe_allow_html=True)
#st.write(f"On a scale of 1-100, where do you fall?") 
st.write("")
st.markdown('<div style="text-align: center">Move the slider or click anywhere on the bar to choose your vibe</div>', unsafe_allow_html=True)
st.write("")

def main():
    # Initialize session state
    if 'character' not in st.session_state:
        st.session_state.character = None
    if 'chat' not in st.session_state:
        st.session_state.chat = None
    # Initialize session state variables at the start of your function
    if 'user_input' not in st.session_state:
        st.session_state.user_input = {}
    if 'asked_questions' not in st.session_state:
        st.session_state.asked_questions = set()
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = None
    if 'last_button' not in st.session_state:
        st.session_state.last_button = None
    if 'show_greeting' not in st.session_state:
        st.session_state.show_greeting = True  # Flag to control whether to show the greeting message

    # Initialize chat
    chat = None
    
    # Create a dictionary to store the user input
    user_input = {}

    # Create a set to store the asked questions
    asked_questions = set()

    # Initialize character as an empty DataFrame
    character = pd.DataFrame()

    # For each selected feature
    for feature in selected_features:
        # Find the corresponding question
        matching_questions = psych_stats_df['question'][psych_stats_df['question'].str.contains(feature, regex=False)].values
        if matching_questions.size > 0:
            question = matching_questions[0]
        else:
            st.write(f"No matching question found for feature {feature}. Skipping this feature.")
            continue

        # Add the question to the set of asked questions
        st.session_state.asked_questions.add(question)

        # Split the question into two parts
        parts = question.split('/')
        
        # Now define your columns 
        col4, col5, col6 = st.columns([1, 5, 1])

        with col4:
            st.write(f"<div class='lower-entry'>{parts[0]}</div>", unsafe_allow_html=True)

        with col5:
            # Center the text using HTML and CSS
            st.write("")
            # Add custom CSS to remove empty space above sliders
            answer = st.slider("", 1, 100, 50, key=f"{feature}_slider")
            st.write("")

        with col6:
            st.write(f"<div class='lower-entry'>{parts[1]}</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <style>
            .lower-entry {
                margin-top: 50px; /* Adjust the margin-top value to move the text lower */
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Calculate the score for the selected feature
        if feature == question.split('/')[0]:
            score = 100 - answer if answer <= 50 else 0
        else:
            score = answer if answer > 50 else 0

        st.session_state.user_input[feature] = score

    # Universe recommender
    if selected_universe == 'Find my universe instead':
        if st.button('Find my universe'):
            # Convert the user input to a list in the same order as the selected features
            user_input_df = pd.DataFrame([user_input], columns=selected_features)

            # Fill NaN values with 0
            user_input_df = user_input_df.fillna(0)

            # Use the model to find the nearest neighbor
            _, indices = nn_model.kneighbors(user_input_df)
            nearest_universe = matrix_df.iloc[indices[0][0]].name
            st.write('You should live in:')
            st.write(nearest_universe)

     # Character recommender
    else:
        if st.button('Find my character', key="find_character_button"):
            # Convert the user input to a list in the same order as the selected features
            user_input_df = pd.DataFrame([st.session_state.user_input], columns=selected_features)

            # Fill NaN values with 0
            user_input_df = user_input_df.fillna(0)

            # Use the model to find the nearest neighbor
            _, indices = nn_model.kneighbors(user_input_df)

            # Get the char_id of the nearest neighbor
            nearest_char_id = matrix_df.iloc[indices[0][0]].name

            # Find the row in characters_df that matches the char_id
            character = characters_df.loc[characters_df['id'] == nearest_char_id, ['name', 'uni_name', 'link', 'wiki_link', 'image_link']]

            st.session_state.character = character
            chat, chat_response = chatbot_init()  # Update to get the initial chat response

            # Display the greeting message from the chatbot
            st.write(chat_response)

        # Display character information
        if st.session_state.character is not None:
            # Load the image from the 'image_link' column
            image_link = st.session_state.character['image_link'].values[0]
            st.image(image_link, caption=st.session_state.character['name'].values[0])

            # Print the other character information
            st.write('Name:', st.session_state.character['name'].values[0])
            st.write('Universe:', st.session_state.character['uni_name'].values[0])
            st.write('Link:', st.session_state.character['link'].values[0])
            st.write('Wikipedia link:', st.session_state.character['wiki_link'].values[0])    # Added Wikipedia link

            # Only show the greeting message once
            st.session_state.show_greeting = False

            # Initialize chat if not done already
            if chat is None:
                chat, chat_response = chatbot_init()  # Update to get the initial chat response

            # Get user message
            with st.form(key='chat_form'):
                user_message = st.text_input('Your message:', key="user_message")

                # Process user message when the form is submitted
                if st.form_submit_button('Send') or st.session_state.button_clicked == 'send_button':
                    if user_message.lower() == 'exit' or user_message.lower() == 'quit':
                        st.write("Ending chat session.")
                    elif user_message:
                        chat_response = get_chat_response(chat, user_message)
                        st.write(chat_response)

                # Store the last button clicked
                st.session_state.last_button = st.session_state.button_clicked
                st.session_state.button_clicked = None

            # Add JavaScript to focus on text input field when the page is loaded
            st.write("""
            <script>
                document.addEventListener('DOMContentLoaded', function() {
                    document.querySelector('.stTextInput > div > div > input').focus();
                });
            </script>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()