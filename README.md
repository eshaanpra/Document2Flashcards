
# Document2Flashcards


This is a streamlit application that creates flashcards from documents using OCR and AI.

The application includes document processing for multiple file formats, AI-powered flashcard generation using the Llama model, and a user authentication system. Users can save their flashcard decks to a database and access them later. The interface includes an interactive flashcard display with a flipping animation.

The application uses Tesseract OCR for image processing and PyPDF2 for PDF text extraction. The Llama model processes text in chunks to generate flashcards in JSON format. A SQLite database stores user accounts, decks, and individual flashcards. The interface is built with Streamlit and includes custom CSS for styling.
## Acknowledgements

 - [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
 - [Hugging Face Training Data](https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf)
 - [Pytesseract](https://github.com/h/pytesseract)
 


## Appendix

The application includes a database schema with tables for users, decks, and flashcards. The flashcard generation process uses a chunk-based approach to handle large documents, with each chunk processed separately by the Llama model. The application also includes session state management to track user authentication status and current flashcard state. The interface features custom CSS for styling the flashcard display, including a 3D flipping animation when users toggle between question and answer views. The sidebar provides access to saved decks, allowing users to quickly switch between different sets of flashcards. The application supports multiple file formats, including PDFs, images (using OCR), and plain text files, making it versatile for different types of educational content.


## Authors

- [@eshaanpra](https://www.github.com/eshaanpra)


## Badges

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

Link to demo video:
https://www.youtube.com/watch?v=veV4jojK0yU
## Deployment

* Document Flashcard Generator
A Streamlit application that creates flashcards from documents using OCR and AI.

# Features

* Document processing for PDFs, images, and text files
* AI-powered flashcard generation using Llama model
* User authentication system with login and signup
* Database storage for saving and retrieving flashcard decks
* Interactive flashcard interface with flipping animation

```bash
pip install -r requirements.txt
```
Install Tesseract OCR:
```bash
# Windows
# Download and install from https://github.com/UB-Mannheim/tesseract/wiki

# macOS
brew install tesseract

# Linux
sudo apt-get install tesseract-ocr ```

Download the Llama model:

```bash
# Download the Phi-3-mini-4k-instruct-q4.gguf model
# Place it in the project directory
```
To run the application locally:

```bash
streamlit run FullProgram.py
```

# Deployment to Streamlit Cloud:

Push code to GitHub:

```bash
# Initialize git repository (if not already done)
git init

# Add all files to git
git add .

# Commit your changes
git commit -m "Initial commit"

# Add your GitHub repository as remote
git remote add origin https://github.com/yourusername/document-flashcard-generator.git

# Push to GitHub
git push -u origin main
```

Deploy to Streamlit Cloud:

* Go to share.streamlit.io in your browser
* Sign in with your GitHub account
* Click "New app"
* Select your repository, branch, and main file (FullProgram.py)
* Click "Deploy"

Configure App on Streamlit Cloud:

```bash
# In the Streamlit Cloud dashboard, go to your app settings
# Add the following secrets:
TESSERACT_PATH=/usr/bin/tesseract
MODEL_PATH=/path/to/your/model/Phi-3-mini-4k-instruct-q4.gguf
```

Update your code to use environment variables:

```bash
# In FullProgram.py, update these lines:
import os

# Use environment variables for paths
pytesseract.pytesseract.tesseract_cmd = os.environ.get('TESSERACT_PATH', r'C:\Program Files\Tesseract-OCR\tesseract.exe')

# In the load_llm function
@st.cache_resource
def load_llm():
    model_path = os.environ.get('MODEL_PATH', "./Phi-3-mini-4k-instruct-q4.gguf")
    return Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=8,
        n_gpu_layers=35
    )
```

Redeploy your app:

```bash
# Make changes to your code
git add .
git commit -m "Update code to use environment variables"
git push origin main

# Streamlit Cloud will automatically redeploy your app
```

Demo Access:
* Username: Demo
* Password: pasword123
# Documentation

* [Streamlit Documentation](https://docs.streamlit.io/)
* [Pandas Documentation](https://pandas.pydata.org/docs/)
* [NumPy Documentation](https://numpy.org/doc/)
* [Pytesseract Documentation](https://pypi.org/project/pytesseract/)
* [Pillow (PIL) Documentation](https://pillow.readthedocs.io/)
* [PyPDF2 Documentation](https://pypdf2.readthedocs.io/)
* [Llama-cpp-python Documentation](https://llama-cpp-python.readthedocs.io/)
* [Streamlit Authenticator Documentation](https://github.com/mkhorasani/streamlit-authenticator)
* [PyYAML Documentation](https://pyyaml.org/wiki/PyYAMLDocumentation)
* [Python-dotenv Documentation](https://saurabh-kumar.com/python-dotenv/)
* [Tesseract OCR Documentation](https://tesseract-ocr.github.io/tessdoc/)
* [SQLite Documentation](https://sqlite.org/docs.html)
* [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
* [Git Documentation](https://git-scm.com/doc)
* [GitHub Documentation](https://docs.github.com/)
* [Python Documentation](https://docs.python.org/3/)
* [Markdown Documentation](https://www.markdownguide.org/basic-syntax/)
* [CSS Documentation](https://developer.mozilla.org/en-US/docs/Web/CSS)
* [HTML Documentation](https://developer.mozilla.org/en-US/docs/Web/HTML)
* [JSON Documentation](https://www.json.org/json-en.html)
* [Python Virtual Environment Documentation](https://docs.python.org/3/tutorial/venv.html)
* [Pip Documentation](https://pip.pypa.io/en/stable/)
