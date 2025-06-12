import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    print("*****Starting...")
    loader = TextLoader(file_path="blog.txt", encoding="utf-8")
    document = loader.load()
    print("\n*****Splitting")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"document split into {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))

    print("\n*****Ingestion")
    PineconeVectorStore.from_documents(
        documents=texts,
        embedding=embeddings,
        index_name=os.environ.get("PINECONE_INDEX_NAME"),
    )

    print("\n*****Finished")
