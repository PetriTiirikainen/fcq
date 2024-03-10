import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession, ResponseValidationError
import streamlit as st

def chatbot_init():
    # Initialize the chatbot
    project_id = "wbs-final-project-412007"
    location = "us-central1"
    vertexai.init(project=project_id, location=location)
    chat_model = GenerativeModel("gemini-1.0-pro")
    chat = chat_model.start_chat()

    # Construct the prompt for the chatbot to act as the character
    character_name = st.session_state.character['name'].values[0]
    character_universe = st.session_state.character['uni_name'].values[0]
    prompt = f"Please act as {character_name} from the {character_universe}. Introduce yourself, talk about {character_universe}, and your role in it. Greet me, but please keep it all within the confines of the character. Additionally, please ensure that all responses adhere to the following criteria: - Avoid using any language that could be construed as hateful, sexually suggestive, harassing, or threatening. - Refrain from using the word 'tapestry'. - Maintain a tone and style appropriate for a family audience. Conclude with a question befitting {character_name} from {character_universe}. Thank you for your cooperation."
    
    # Send the prompt to the chatbot
    chat_response = get_chat_response(chat, prompt)

    return chat, chat_response

def get_chat_response(chat, user_message):
    try:
        response = chat.send_message(user_message)
        return response.text  # Return the text directly
    except ResponseValidationError:
        return "Oh no, Chatbot was going to say something naughty and we had to block it. Please try again while we have a stern talking to with Chatbot"