# Document Processing and Embedding with AstraDB

This repository contains two Python scripts for processing documents and generating embeddings using OpenAI's API, followed by storing the embeddings in an AstraDB collection. The scripts provide a solution for managing text data, including loading various document formats, normalizing the text, generating embeddings, and storing the processed data in a vector database.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Code Overview](#code-overview)
- [Dependencies](#dependencies)
- [Contributing](#contributing)


## Installation

### Clone the repository:

```bash
git clone https://github.com/yourusername/document-embedding-astradb.git
cd document-embedding-astradb
```
# Create a virtual environment:
```bash

python3 -m venv venv
source venv/bin/activate
```
# Install the required packages:
```bash
pip install -r requirements.txt
```
# Download the necessary NLTK data:
```bash
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```
# Usage
Code 1: process_documents_v1.py
This script is designed to:

Load documents (PDF, DOCX, TXT).
Normalize and preprocess text.
Split text into manageable chunks.
Generate embeddings using OpenAI's API.
Store the embeddings and metadata in an AstraDB collection.
Run the script:
Update the file_paths list in the script to include the paths to your documents.
Run the script:
```bash
python process_documents_v1.py
```
# Code 2: process_documents_v2.py
This script offers similar functionality with slight variations, such as:

Adjusted error handling.
Additional emphasis on document references.
Slightly different configurations for chunk size and overlap during text splitting.
Run the script:
Update the file_paths list in the script to include the paths to your documents.
# Run the script:
``` bash
python process_documents_v2.py
```
# Configuration
You need to set up environment variables for your API keys and database tokens. These can be stored securely using google.colab.userdata, or you can use a .env file.

# Required environment variables:
```bash
ASTRA_DB_APPLICATION_TOKEN
ASTRA_DB_API_ENDPOINT
OPENAI_API_KEY
Example configuration:
python
Copy code
import os
from google.colab import userdata

ASTRA_DB_APPLICATION_TOKEN = userdata.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = userdata.get("ASTRA_DB_API_ENDPOINT")
OPENAI_API_KEY = userdata.get("OPENAI_API_KEY")
```
# Code Overview
Document Processing
# DocumentProcessor: Handles the loading of documents from various formats (PDF, DOCX, TXT).
RecursiveCharacterTextSplitterComponent: Splits large text into smaller chunks for efficient processing.
OpenAIEmbeddingsComponent: Generates embeddings for text chunks using OpenAI's API.
# Database Management
AstraDBManager: Manages the interaction with AstraDB, including creating collections and storing documents.
# Main Function
The main() function orchestrates the loading, processing, embedding, and storing of documents.

# Dependencies
```bash
langchain
nltk
astrapy
openai
python-dotenv
```
# Install these dependencies using:

```bash
pip install langchain nltk astrapy openai python-dotenv
```
# Contributing
Contributions are welcome! Please open an issue or submit a pull request with any enhancements or bug fixes.

