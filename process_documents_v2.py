import os
import json
import uuid
from pathlib import Path
from typing import Any, List, Tuple, Optional
from dataclasses import dataclass
from langchain.vectorstores import VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from astrapy import DataAPIClient
import re
from google.colab import userdata
from astrapy.constants import VectorMetric
from astrapy.database import Database
from astrapy.collection import Collection
from astrapy.exceptions import CollectionAlreadyExistsException
import time
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Constants and configurations
TEXT_FILE_TYPES = ["txt", "docx", "pdf"]
ASTRA_DB_APPLICATION_TOKEN = userdata.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = userdata.get("ASTRA_DB_API_ENDPOINT")
OPENAI_API_KEY = userdata.get("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-3-small"

@dataclass
class Data:
    content: str = ""
    metadata: dict = None

def advanced_normalize(text: str) -> str:
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

class DocumentProcessor:
    def __init__(self, path: str, silent_errors: bool = False):
        self.path = path
        self.silent_errors = silent_errors
        self.status = ""

    def resolve_path(self, path: str) -> str:
        return os.path.abspath(path)

    def load_file(self) -> Tuple[Data, str]:
        if not self.path:
            raise ValueError("Please, upload a file to use this component.")
        resolved_path = self.resolve_path(self.path)

        extension = Path(resolved_path).suffix[1:].lower()
        if extension not in TEXT_FILE_TYPES:
            raise ValueError(f"Unsupported file type: {extension}")

        if extension == "docx":
            loader = Docx2txtLoader(resolved_path)
        elif extension == "pdf":
            loader = PyPDFLoader(resolved_path)
        else:  # Treat as text file
            loader = TextLoader(resolved_path)

        data_list = loader.load()

        if isinstance(data_list, list) and len(data_list) > 0:
            content = data_list[0].page_content
            metadata = data_list[0].metadata
            metadata['file_name'] = Path(resolved_path).stem
            return Data(content=content, metadata=metadata), Path(resolved_path).stem
        else:
            return Data(), ""

class RecursiveCharacterTextSplitterComponent:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: Optional[List[str]] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators if separators else [".", "\n", "\n\n"]

    def split_text(self, text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            separators=self.separators,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.split_text(text)

class OpenAIEmbeddingsComponent:
    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model_name

    def build_embeddings(self, texts: List[str]) -> List[List[float]]:
        embedding_model = OpenAIEmbeddings(
            model=self.model,
            api_key=self.api_key
        )
        embeddings = []
        batch_size = 10
        max_tokens = 1000

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                total_tokens = sum(len(text.split()) for text in batch)
                if total_tokens > max_tokens:
                    print(f"Warning: Batch size exceeds token limit with {total_tokens} tokens.")

                embeddings.extend(embedding_model.embed_documents(batch))
                time.sleep(60)  # Rate limit management

            except Exception as e:
                print(f"Error during embedding: {e}. Retrying after 60 seconds...")
                time.sleep(60)

        return embeddings

class AstraDBManager:
    def __init__(self, api_endpoint: str, token: str, collection_name: str, namespace: Optional[str] = None):
        self.api_endpoint = api_endpoint
        self.token = token
        self.collection_name = collection_name
        self.namespace = namespace or "default_namespace"
        self.client = DataAPIClient(token)
        self.database = self.client.get_database(api_endpoint)

    def get_or_create_collection(self, collection_name: str, dimension: int = 1536):
        try:
            print(f"Checking for collection {collection_name} in database.")
            collections = self.database.list_collections()
            if collection_name in collections:
                print(f"* Collection {collection_name} already exists.")
                return self.database.get_collection(collection_name)
            else:
                print(f"* Collection {collection_name} does not exist. Creating...")
                collection = self.database.create_collection(
                    name=collection_name,
                    dimension=dimension,
                    metric=VectorMetric.COSINE,
                )
                print(f"* Collection {collection_name} created successfully.")
                return collection
        except CollectionAlreadyExistsException:
            print(f"* Collection {collection_name} already exists. Skipping creation.")
            return self.database.get_collection(collection_name)
        except Exception as e:
            print(f"Error handling collection {collection_name}: {e}")
            raise

    def add_documents(self, embeddings: List[List[float]], doc_name: str, metadata: dict, chunks: List[str]):
        collection = self.get_or_create_collection(collection_name=self.collection_name)

        for index, embedding in enumerate(embeddings):
            doc_id = str(uuid.uuid4())
            document = {
                "_id": doc_id,
                "content": chunks[index],
                "$vector": embedding,
                "metadata": {
                    "doc_name": doc_name,
                    **metadata
                }
            }
            try:
                result = collection.insert_one(document)
                if result.inserted_id:
                    print(f"Inserted document {doc_id}")
                else:
                    print(f"Failed to insert document {doc_id}")
            except Exception as e:
                print(f"Error processing document {doc_id}: {e}")

def main(file_paths: List[str]):
    text_splitter = RecursiveCharacterTextSplitterComponent(chunk_size=500, chunk_overlap=50)
    embeddings_component = OpenAIEmbeddingsComponent(api_key=OPENAI_API_KEY)
    astradb_manager = AstraDBManager(
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
        collection_name="construction_handbooks"
    )

    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        processor = DocumentProcessor(file_path)

        # Load and process file
        data, doc_name = processor.load_file()
        
        # Normalize content
        normalized_content = advanced_normalize(data.content)
        
        chunks = text_splitter.split_text(normalized_content)

        # Embeddings
        embeddings = embeddings_component.build_embeddings(chunks)

        # Send to AstraDB
        astradb_manager.add_documents(embeddings, doc_name, data.metadata, chunks)
        print(f"Finished processing file: {file_path}")

if __name__ == "__main__":
    file_paths = [
        "",
        # Add more handbook paths here
    ]
    main(file_paths)
