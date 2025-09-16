from langchain_community.document_loaders import PyMuPDFLoader,DirectoryLoader
from  langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from typing import List
from langchain.schema import Document

#Extract Data from the pdf
def load_pdf_file(data):
    loader=DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyMuPDFLoader
    )
    documents=loader.load()
    return documents

def filter_to_minimal_docs(docs:List[Document])->List[Document]:
    minimal_docs:List[Document]=[]
    for doc in docs:
        src=doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source":src}
            )
        )
    return minimal_docs

#split the data into chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

#download the embeddings from the hugging face
def download_embedding():
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

