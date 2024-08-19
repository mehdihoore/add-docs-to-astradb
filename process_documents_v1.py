import os
import json
import uuid
from pathlib import Path
from typing import Any, List, Tuple, Optional
from dataclasses import dataclass
from langchain.vectorstores import VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import Docx2txtLoader  # Updated import path for DocxLoader
from langchain.document_loaders import PyPDFLoader  # Updated import path for PDFLoader
from langchain.document_loaders import TextLoader
from astrapy import DataAPIClient
import re
from google.colab import userdata
from astrapy.constants import VectorMetric
from astrapy.database import Database
from astrapy.collection import Collection
from astrapy.exceptions import CollectionAlreadyExistsException
import time

# Constants and configurations
TEXT_FILE_TYPES = ["txt", "docx", "pdf"]
ASTRA_DB_APPLICATION_TOKEN = userdata.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = userdata.get("ASTRA_DB_API_ENDPOINT")
OPENAI_API_KEY = userdata.get("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-3-small"  # Change if needed

@dataclass
class Data:
    content: str = ""
    references: str = ""

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

        # Ensure data_list contains the expected content
        if isinstance(data_list, list) and len(data_list) > 0:
            # Assuming the content of the first item in the list
            data = Data(content=data_list[0].page_content)
            return data, Path(resolved_path).stem
        else:
            return Data(), ""

class RecursiveCharacterTextSplitterComponent:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: Optional[List[str]] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators if separators else [".", "\n"]

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
        batch_size = 10  # Adjust batch size as needed
        max_tokens = 1000  # Ensure token limits per batch

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                # Optionally, calculate token count here
                total_tokens = sum(len(text.split()) for text in batch)
                if total_tokens > max_tokens:
                    print(f"Warning: Batch size exceeds token limit with {total_tokens} tokens.")

                embeddings.extend(embedding_model.embed_documents(batch))
                time.sleep(60)  # Introduce delay to manage rate limits

            except RateLimitError as e:
                print(f"Rate limit error: {e}. Retrying after 60 seconds...")
                time.sleep(60)  # Wait before retrying

            except OpenAIError as e:
                print(f"OpenAI API error: {e}")
                time.sleep(60)  # Delay before retrying for other OpenAI errors

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

    def add_documents(self, embeddings: List[List[float]], doc_name: str, references: str, chunks: List[str]):
        collection = self.get_or_create_collection(collection_name=self.collection_name)

        for index, embedding in enumerate(embeddings):
            doc_id = str(uuid.uuid4())
            document = {
                "_id": doc_id,
                "content": chunks[index],
                "$vector": embedding,
                "metadata": {
                    "doc_name": doc_name,
                    "references": references,
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


def extract_references(text: str) -> str:
    # Implement logic to extract references in the format "2-8-7-22"
    pattern = r'\d+-\d+-\d+-\d+'
    matches = re.findall(pattern, text)
    return matches[0] if matches else "No references"

def main(file_paths: List[str]):
    # Initialize components
    text_splitter = RecursiveCharacterTextSplitterComponent()
    embeddings_component = OpenAIEmbeddingsComponent(api_key=OPENAI_API_KEY)
    astradb_manager = AstraDBManager(
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
        collection_name="nashriat"
    )

    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        processor = DocumentProcessor(file_path)

        # Load and process file
        data, doc_name = processor.load_file()
        chunks = text_splitter.split_text(data.content)

        # Embeddings
        embeddings = embeddings_component.build_embeddings(chunks)

        # Extract references from the whole document
        references = extract_references(data.content)

        # Send to AstraDB
        astradb_manager.add_documents(embeddings, doc_name, references, chunks)
        print(f"Finished processing file: {file_path}")

if __name__ == "__main__":
    file_paths = [
        "",
        "",
        "",
        "",


    ]
    main(file_paths)
