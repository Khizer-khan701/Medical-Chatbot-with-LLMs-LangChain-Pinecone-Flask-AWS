from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_embedding

# Load keys
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Load & preprocess documents
extracted_data = load_pdf_file("data/")
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)
embeddings = download_embedding()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws",region="us-east-1")
    )
index = pc.Index(index_name)

# Load index
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,   # use once for ingestion
    embedding=embeddings,
    index_name=index_name
)

print("Data successfully ingested into pinecone")
