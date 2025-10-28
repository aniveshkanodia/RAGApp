from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

def load():
    #Load the PDF Document
    loader = PyPDFLoader("/Users/akanodia/RAGApp/RAG_App/documents/AI in 2024.pdf")
    pages = loader.load_and_split()
    '''
    loader.load will load the entire pdf in 1 document object.
    load_and_split create separate document object for every split.
    '''
    #split pages by char
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100,
        length_function = len,
        add_start_index = True
    )
    chunks = text_splitter.split_documents(pages)
    print(f"Split {len(pages)} documents into {len(chunks)} chunks.")
    return chunks
document_chunks = load()

#Initialise the models 

#ollama pull qwen3-embeddings
embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

#Initialise vector store
vectordb = Chroma(collection_name="documents", embedding_function = embeddings, persist_directory="./db/chroma_db") #path to save data locally

#store documents in vectordb
document_ids = vectordb.add_documents(documents=document_chunks)


#Define the retrieval chain
retriever = vectordb.as_retriever(search_type = "similarity",
                                  search_kwargs={
                                        "k":4
                                  })


#ollama pull qwen3
llm = ChatOllama(model="qwen3:0.6b",keep_alive="2h", temperature=0)


#Define the chat prompt
chat_prompt = ChatPromptTemplate(
    [
        ("system", "You are a helpful assistant. Use the context to answer the question. If you don’t know, say you don’t know."),
        ("human", "{context}\n\nQuestion: {question}")
    ]
)


#Retrieve context

retrieve_context = RunnableLambda(
    lambda input_dict: {
        "context": "\n\n".join(
            doc.page_content for doc in retriever.invoke(input_dict["question"])
        ),
        "question": input_dict["question"],
    }
)

# add output parser to pipeline to extract answer only
output_parser = StrOutputParser()

#create pipeline
pipeline_with_parser = retrieve_context | chat_prompt | llm | output_parser

# --- Traceable pipeline execution ---
@traceable(name="RAGApp")
def run_pipeline(question: str):
    return pipeline_with_parser.invoke({"question": question})

if __name__ == "__main__":
    result = run_pipeline("What challenges exist regarding AI adoption?")
    print(result)

#Execute the pipeline with output parser
'''
input_data = {
    "question": "What challenges exist regarding AI adoption?"}
result = pipeline_with_parser.invoke(input_data)
print(result)
'''


