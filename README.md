# Streamlit HF Chat App

A minimal Streamlit chat app that uses the Hugging Face Transformers library to chat with a causal/decoder model. The app accepts a model identifier in the `repo/model` format and provides a chat UI.

Features
- Enter any Hugging Face model ID (public or private).
- Simple chat UI with conversation history.
- Option to provide a Hugging Face token for access to private models.

Security and notes
- Large models require a GPU and plenty of RAM; running on CPU may be very slow.
- If you want to use private HF models, obtain a Hugging Face token and enter it in the app or set it in the environment variable `HF_TOKEN`.
- This repository can be made private when you create it on GitHub (see steps below).

How to run locally
1. Create a Python venv and install dependencies:
   - python -m venv .venv
   - source .venv/bin/activate (or .venv\Scripts\activate on Windows)
   - pip install -r requirements.txt

2. Run the app:
   - streamlit run app.py

3. If using a private HF repo, set your token:
   - export HF_TOKEN="hf_xxx"  (Linux/macOS)
   - set HF_TOKEN="hf_xxx"  (Windows PowerShell)

Making the GitHub repo private and pushing (using GitHub CLI)
1. Initialize git and create repo (private) and push:
   - git init
   - git add .
   - git commit -m "Initial commit: Streamlit HF chat app"
   - gh repo create <owner>/<repo> --private --source=. --remote=origin --push

Or create a new private repo through GitHub web UI and then push to it.

Files in this repo
- app.py: Streamlit app
- requirements.txt: pip dependencies
- .gitignore

