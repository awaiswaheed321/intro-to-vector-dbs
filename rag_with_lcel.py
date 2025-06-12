import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    print("Retrieving...")

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-4o-mini")

    query = "What is use of vector databases?"

    vectorstore = PineconeVectorStore(
        index_name=os.environ.get("PINECONE_INDEX_NAME"), embedding=embeddings
    )

    template = """
    Use the following pieces of context to answer the question at the end. 
    If you dot know the answer, just say that you dont know. Dont try to make up an answer.
    use three sentences maximum and keep answer as concise as possible.
    always say "thanks for asking!" at the end of the answer.
    
    {context}
    
    Question: {question}
    
    Helpful Answer:
    """

    custom_rag_prompt = PromptTemplate.from_template(template=template)

    rag_chain = (
        {
            "context": vectorstore.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
        }
        | custom_rag_prompt
        | llm
    )

    res = rag_chain.invoke(query)

    print(res.content)
