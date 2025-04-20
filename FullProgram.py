import streamlit as st
import pandas as pd
import numpy as np
import sqlite3 as sql
import pytesseract 
from PIL import Image
import PyPDF2 as pdfread
import io
from llama_cpp import Llama
import json
import random
import time
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
import datetime
# Set up Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize database for storing flashcard decks
def init_db():
    conn = sql.connect('flashcards.db')
    c = conn.cursor()
    
    # Create tables if they don't exist (without dropping existing ones)
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS decks
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT, 
                  deck_name TEXT, 
                  created_date TEXT,
                  FOREIGN KEY (username) REFERENCES users(username))''')
    c.execute('''CREATE TABLE IF NOT EXISTS flashcards
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  deck_id INTEGER, 
                  question TEXT, 
                  answer TEXT,
                  FOREIGN KEY (deck_id) REFERENCES decks(id))''')
    conn.commit()
    conn.close()

# Initialize the database
init_db()

# Load Llama Model (cache it to optimize performance)
@st.cache_resource
def load_llm():
    return Llama(
        model_path="./Phi-3-mini-4k-instruct-q4.gguf",
        n_ctx=4096,
        n_threads=8,
        n_gpu_layers=35
    )

# Function to generate flashcards from text using the Llama model
@st.cache_data
def generate_flashcards(text, _llm):
    if not text or len(text.strip()) == 0:
        return []
    
    prompt = """
    Based on the following text, generate 5-10 flashcards based on the amount of content in JSON format.
    Each flashcard should have a "question" and "answer" field.
    The questions should test understanding of key concepts from the text.
    Format the response as a valid JSON array of objects.
    Be creative with the questions.
    
    Example format:
    [
        {"question": "What is the main idea of the text?", "answer": "The main idea is..."},
        {"question": "Define the term X", "answer": "X is defined as..." }
        {"question": "What is true for X", "answer": "Y is true when X..."}
    ]
    
    Text to generate flashcards from:
    """
    
    chunk_size = 1000
    text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    all_flashcards = []
    progress_bar = st.progress(0)
    
    # Loop through chunks of the text to get flashcards for each
    for i, chunk in enumerate(text_chunks):
        with st.spinner("Reading text..."):
            llm = load_llm()
        progress_bar.progress((i + 1) / len(text_chunks))
        
        full_prompt = f"<|user|>\n{prompt}\n{chunk}\n<|end|>\n<|assistant|>"
        
        output = llm(
            prompt=full_prompt,
            max_tokens=1024,
            stop=["<|end|>"],
            echo=False
        )
        
        response_text = output['choices'][0]['text']
        
        try:
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                flashcards = json.loads(json_str)
                all_flashcards.extend(flashcards)
        except Exception as e:
            st.warning(f"Could not parse flashcards from chunk {i+1}: {str(e)}")
    
    return all_flashcards

# Function to clean and optimize text before processing
def optimize_text(text):
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove very short lines (likely noise)
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if len(line.strip()) > 10]
    
    # Join back with newlines
    return '\n'.join(cleaned_lines)

# Function to save a deck of flashcards to the database
def save_deck(username, deck_name, flashcards):
    conn = sql.connect('flashcards.db')
    c = conn.cursor()
    
    # Insert the deck
    created_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO decks (username, deck_name, created_date) VALUES (?, ?, ?)",
              (username, deck_name, created_date))
    deck_id = c.lastrowid
    
    # Insert all flashcards
    for card in flashcards:
        c.execute("INSERT INTO flashcards (deck_id, question, answer) VALUES (?, ?, ?)",
                  (deck_id, card["question"], card["answer"]))
    
    conn.commit()
    conn.close()
    return deck_id

# Function to get all decks for a user
def get_user_decks(username):
    conn = sql.connect('flashcards.db')
    c = conn.cursor()
    c.execute("SELECT id, deck_name, created_date FROM decks WHERE username = ? ORDER BY created_date DESC",
              (username,))
    decks = c.fetchall()
    conn.close()
    return decks

# Function to get all flashcards in a deck
def get_deck_flashcards(deck_id):
    conn = sql.connect('flashcards.db')
    c = conn.cursor()
    c.execute("SELECT question, answer FROM flashcards WHERE deck_id = ?", (deck_id,))
    cards = c.fetchall()
    conn.close()
    
    # Convert to the format expected by the app
    flashcards = []
    for card in cards:
        flashcards.append({"question": card[0], "answer": card[1]})
    return flashcards

# Function to delete a deck
def delete_deck(deck_id):
    conn = sql.connect('flashcards.db')
    c = conn.cursor()
    
    # Delete all flashcards in the deck
    c.execute("DELETE FROM flashcards WHERE deck_id = ?", (deck_id,))
    
    # Delete the deck
    c.execute("DELETE FROM decks WHERE id = ?", (deck_id,))
    
    conn.commit()
    conn.close()

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .auth-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        max-width: 500px;
        margin: 0 auto;
    }
    .auth-button {
        background-color: #1E88E5;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: bold;
        cursor: pointer;
        transition: background-color 0.3s;
        width: 100%;
        margin: 0.5rem 0;
    }
    .auth-button:hover {
        background-color: #1565C0;
    }
    .auth-divider {
        display: flex;
        align-items: center;
        text-align: center;
        margin: 1.5rem 0;
    }
    .auth-divider::before,
    .auth-divider::after {
        content: '';
        flex: 1;
        border-bottom: 1px solid #dee2e6;
    }
    .auth-divider span {
        padding: 0 1rem;
        color: #6c757d;
    }
    .flashcard-container {
        perspective: 1000px;
        width: 100%;
        height: 300px;
        margin: 20px 0;
    }
    .flashcard {
        position: relative;
        width: 100%;
        height: 100%;
        text-align: center;
        transition: transform 0.8s;
        transform-style: preserve-3d;
        cursor: pointer;
    }
    .flashcard.flipped {
        transform: rotateY(180deg);
    }
    .flashcard-front, .flashcard-back {
        position: absolute;
        width: 100%;
        height: 100%;
        backface-visibility: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .flashcard-front {
        background-color: #f8f9fa;
        border: 2px solid #dee2e6;
    }
    .flashcard-back {
        background-color: #e9ecef;
        border: 2px solid #dee2e6;
        transform: rotateY(180deg);
    }
    .flashcard-content {
        font-size: 1.2rem;
        max-width: 100%;
        overflow-wrap: break-word;
    }
    .flashcard-controls {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 20px;
    }
    .flashcard-nav {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-top: 20px;
    }
    .button-container {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-top: 20px;
    }
    .nav-button {
        background-color: #1E88E5;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .nav-button:hover {
        background-color: #1565C0;
    }
    .random-button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        cursor: pointer;
        transition: background-color 0.3s;
        width: 100%;
        max-width: 400px;
        margin: 10px auto;
        display: block;
    }
    .random-button:hover {
        background-color: #388E3C;
    }
    .deck-item {
        padding: 10px;
        margin-bottom: 5px;
        border-radius: 5px;
        background-color: #f8f9fa;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .deck-item:hover {
        background-color: #e9ecef;
    }
    .deck-item.active {
        background-color: #1E88E5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'users' not in st.session_state:
    # Initialize with some demo users (in a real app, this would be in a database)
    st.session_state.users = {
        "demo": "password123",
        "admin": "admin123"
    }
if 'current_deck_id' not in st.session_state:
    st.session_state.current_deck_id = None
if 'deck_name' not in st.session_state:
    st.session_state.deck_name = ""

# Authentication page
if not st.session_state.authenticated:
    st.markdown("<h1 class='main-header'>Document Flashcard Generator</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='auth-container'>", unsafe_allow_html=True)
    
    # Login form
    st.subheader("Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    
    if st.button("Login", key="login_button", use_container_width=True):
        # Proper login validation
        if username and password:
            if username in st.session_state.users and st.session_state.users[username] == password:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success(f"Welcome back, {username}!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Invalid username or password")
        else:
            st.error("Please enter both username and password")
    
    st.markdown("<div class='auth-divider'><span>OR</span></div>", unsafe_allow_html=True)
    
    # Signup form
    st.subheader("Sign Up")
    new_username = st.text_input("Choose Username", key="signup_username")
    new_password = st.text_input("Choose Password", type="password", key="signup_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
    
    if st.button("Sign Up", key="signup_button", use_container_width=True):
        # Proper signup validation
        if new_username and new_password and confirm_password:
            if new_password == confirm_password:
                if new_username in st.session_state.users:
                    st.error("Username already exists. Please choose another.")
                else:
                    # Add new user to the users dictionary
                    st.session_state.users[new_username] = new_password
                    st.session_state.authenticated = True
                    st.session_state.username = new_username
                    st.success(f"Account created successfully! Welcome, {new_username}!")
                    time.sleep(1)
                    st.rerun()
            else:
                st.error("Passwords do not match")
        else:
            st.error("Please fill in all fields")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display demo credentials
    with st.expander("Demo Credentials", expanded=False):
        st.write("You can use these demo accounts to log in:")
        st.code("Username: demo\nPassword: password123")
        st.code("Username: admin\nPassword: admin123")
    
    # Stop execution here if not authenticated
    st.stop()

# Main application (only shown after authentication)
st.title("Document Flashcard and Quiz Generator")
st.write(f"Welcome, {st.session_state.username}! Upload one or more documents to extract text and generate flashcards.")

# Sidebar for saved decks
with st.sidebar:
    st.header("Your Flashcard Decks")
    
    # Get user's decks
    user_decks = get_user_decks(st.session_state.username)
    
    if user_decks:
        st.write(f"You have {len(user_decks)} saved decks:")
        for deck_id, deck_name, created_date in user_decks:
            # Create a button for each deck
            if st.button(f"{deck_name} ({created_date.split()[0]})", key=f"deck_{deck_id}", use_container_width=True):
                st.session_state.current_deck_id = deck_id
                st.session_state.flashcards = get_deck_flashcards(deck_id)
                st.session_state.current_flashcard = 0
                st.session_state.show_answer = False
                st.rerun()
        
        # Add a delete button for the current deck
        if st.session_state.current_deck_id:
            if st.button("Delete Current Deck", key="delete_deck", use_container_width=True):
                delete_deck(st.session_state.current_deck_id)
                st.session_state.current_deck_id = None
                st.session_state.flashcards = None
                st.success("Deck deleted successfully!")
                st.rerun()
    else:
        st.info("You don't have any saved decks yet. Create one by generating flashcards and saving them.")

# File uploader to choose documents
uploaded_files = st.file_uploader("Choose one or more files", type=["jpg", "jpeg", "png", "pdf", "txt"], accept_multiple_files=True)

all_text = ""

# Process uploaded files
if uploaded_files: 
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        st.subheader(f"Processing file: {filename}")
        
        # Extract text based on file type
        if uploaded_file.type.startswith("image"):
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            img = Image.open(uploaded_file)
            extracted_text = pytesseract.image_to_string(img)
            
        elif uploaded_file.type == "application/pdf":
            pdf_bytes = uploaded_file.read()
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = pdfread.PdfReader(pdf_file)
            
            extracted_text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                extracted_text += page.extract_text()
                
        elif uploaded_file.type == "text/plain":
            extracted_text = uploaded_file.read().decode("utf-8")
            
        else:
            st.error(f"Error processing {filename}: Unsupported file type")
            continue
        
        all_text += f"\n\n--- Text from {filename} ---\n\n{extracted_text}"
    
    # Show raw extracted text in expandable container
    # with st.expander("View Raw Extracted Text", expanded=False):
    #     st.text(all_text)
    
    # Generate flashcards from extracted text
    if st.button("Generate Flashcards"):
        with st.spinner("Generating flashcards with AI..."):
            llm = load_llm()
            
            flashcards = generate_flashcards(all_text, llm)
            
            # Save flashcards in session state
            if flashcards:
                st.session_state.flashcards = flashcards
                st.session_state.current_flashcard = 0
                st.session_state.show_answer = False
                st.session_state.current_deck_id = None  # Reset current deck ID
                
                st.success(f"Generated {len(flashcards)} flashcards!")
            else:
                st.error("No flashcards could be generated from the text.")

# Display and interact with flashcards
if 'flashcards' in st.session_state and st.session_state.flashcards:
    st.subheader("Flashcards")
    
    # Save deck option
    if st.session_state.current_deck_id is None:  # Only show save option for newly generated decks
        col1, col2 = st.columns([3, 1])
        with col1:
            deck_name = st.text_input("Deck Name", value=st.session_state.deck_name, 
                                     placeholder="Enter a name for this deck")
        with col2:
            if st.button("Save Deck", use_container_width=True):
                if deck_name:
                    deck_id = save_deck(st.session_state.username, deck_name, st.session_state.flashcards)
                    st.session_state.current_deck_id = deck_id  # Set the current deck ID to the newly saved deck
                    st.success(f"Deck '{deck_name}' saved successfully!")
                    st.session_state.deck_name = ""  # Reset deck name
                    st.rerun()
                else:
                    st.error("Please enter a name for the deck")
    
    # Display current flashcard
    current_idx = st.session_state.current_flashcard
    current_flashcard = st.session_state.flashcards[current_idx]
    
    # Create a container for the flashcard
    flashcard_container = st.container()
    
    # Flashcard HTML with animation
    with flashcard_container:
        st.markdown(f"""
        <div class="flashcard-container">
            <div class="flashcard {'flipped' if st.session_state.show_answer else ''}" id="flashcard">
                <div class="flashcard-front">
                    <div class="flashcard-content">
                        <h3>Question {current_idx + 1}/{len(st.session_state.flashcards)}</h3>
                        <p>{current_flashcard["question"]}</p>
                    </div>
                </div>
                <div class="flashcard-back">
                    <div class="flashcard-content">
                        <h3>Answer</h3>
                        <p>{current_flashcard["answer"]}</p>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Flashcard controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Previous", use_container_width=True) and current_idx > 0:
            st.session_state.current_flashcard -= 1
            st.session_state.show_answer = False
            st.rerun()
    
    with col2:
        if st.button("Flip Card", use_container_width=True):
            st.session_state.show_answer = not st.session_state.show_answer
            time.sleep(0.1)  # Small delay for animation
            st.rerun()
    
    with col3:
        if st.button("Next", use_container_width=True) and current_idx < len(st.session_state.flashcards) - 1:
            st.session_state.current_flashcard += 1
            st.session_state.show_answer = False
            st.rerun()
    
    # Random flashcard navigation
    st.markdown("<div style='width: 100%; max-width: 600px; margin: 0 auto;'>", unsafe_allow_html=True)
    if st.button("Random Flashcard", use_container_width=True):
        st.session_state.current_flashcard = random.randint(0, len(st.session_state.flashcards) - 1)
        st.session_state.show_answer = False
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# Logout button at the bottom
if st.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.success("You have been logged out successfully!")
    time.sleep(1)
    st.rerun()