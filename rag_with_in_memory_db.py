import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

if __name__ == "__main__":

    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    loader = PyPDFLoader("paper.pdf")
    document = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    split_docs = text_splitter.split_documents(documents=document)

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(documents=split_docs, embedding=embeddings)
    vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local(
        "faiss_index_react", embeddings, allow_dangerous_deserialization=True
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=new_vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    query = "Give me the GIST of reAct in 3 sentences."

    result = retrieval_chain.invoke(input={"input": query})
    print(f"RAG Result: {result['answer']}")
