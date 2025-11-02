from dotenv import load_dotenv

#load environment variables from a .env file for langsmith tracing
load_dotenv()

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langsmith import traceable
import gradio as gr
import os
from typing import List

def load_document(file_path: str) -> List[Document]:
    """Load a document based on its file extension.
    
    Args:
        file_path: Path to the file to load
        
    Returns:
        List of Document objects from the file
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == ".pdf":
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    elif file_ext == ".txt":
        loader = TextLoader(file_path)
        documents = loader.load()
    elif file_ext in [".docx", ".doc"]:
        loader = UnstructuredWordDocumentLoader(file_path)
        documents = loader.load()
    else:
        raise ValueError(f"Unsupported file type: {file_ext}. Supported types: .pdf, .txt, .docx, .doc")
    
    return documents

embeddings = None
vectordb = None
retriever = None
llm = None
chat_prompt = None
pipeline_with_parser = None


def init_rag():
    global embeddings, vectordb, retriever, llm, chat_prompt, pipeline_with_parser

    if pipeline_with_parser is not None:
        return

    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

    vectordb = Chroma(
        collection_name="documents",
        embedding_function=embeddings,
        persist_directory="./db/chroma_db",
    )

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    llm = ChatOllama(model="qwen3:0.6b", keep_alive="2h", temperature=0)

    chat_prompt = ChatPromptTemplate(
        [
                (
                    "system",
                    "You are a helpful assistant. Use the context to answer the question. If you don't know, say you don't know.",
                ),
                ("human", "{context}\n\nQuestion: {question}"),
        ]
    )

    retrieve_context = RunnableLambda(
        lambda input_dict: {
            "context": "\n\n".join(
                    doc.page_content for doc in retriever.invoke(input_dict["question"])  # type: ignore[misc]
            ),
            "question": input_dict["question"],
        }
    )

    output_parser = StrOutputParser()
    pipeline_with_parser = retrieve_context | chat_prompt | llm | output_parser


# --- Traceable pipeline execution ---
@traceable(name="RAGApp")
def run_pipeline(question: str):
    init_rag()
    return pipeline_with_parser.invoke({"question": question})  # type: ignore[misc]


def answer_question(question: str) -> str:
    return run_pipeline(question)


def process_and_index_file(file_path: str) -> str:
    """Process an uploaded file and add it to the vector database.
    
    Args:
        file_path: Path to the uploaded file
        
    Returns:
        Status message indicating success or error
    """
    if not file_path:
        return "Error: No file provided."
    
    try:
        # Ensure RAG is initialized
        init_rag()
        
        # Ensure vectordb is available
        if vectordb is None:  # type: ignore[misc]
            return "Error: Vector database not initialized."
        
        # Load document
        documents = load_document(file_path)
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True
        )
        chunks = text_splitter.split_documents(documents)
        
        # Add to existing vector database
        vectordb.add_documents(documents=chunks)  # type: ignore[misc]
        
        return f"Successfully uploaded and indexed {len(chunks)} chunks from {os.path.basename(file_path)}"
    except Exception as e:
        return f"Error processing file: {str(e)}"

def build_ui():
    init_rag()

    def on_submit(user_msg, chat_history):
        if not user_msg or not str(user_msg).strip():
            return chat_history, chat_history, ""
        answer = answer_question(str(user_msg))
        chat_history = (chat_history or []) + [(str(user_msg), answer)]
        return chat_history, chat_history, ""
    
    def on_file_upload(file):
        if file is None:
            return "No file uploaded."
        
        status_msg = "Processing..."
        yield status_msg
        
        # Get file path - Gradio File component returns a path string or file object
        file_path = file.name if hasattr(file, 'name') else str(file)
        
        # Process the uploaded file
        result = process_and_index_file(file_path)
        yield result

    with gr.Blocks() as demo:
        gr.Markdown("## RAG Chat")
        
        with gr.Row():
            with gr.Column(scale=2):
                chat = gr.Chatbot(height=500)
                msg = gr.Textbox(placeholder="Ask a question…", label="Your question")
                state = gr.State([])
                clear = gr.Button("Clear")

                msg.submit(on_submit, [msg, state], [chat, state, msg])
                clear.click(lambda: ([], [], ""), None, [chat, state, msg])
            
            with gr.Column(scale=1):
                gr.Markdown("### Upload Document")
                file_upload = gr.File(
                    label="Upload PDF, TXT, or DOCX file",
                    file_types=[".pdf", ".txt", ".docx", ".doc"]
                )
                upload_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    placeholder="No file uploaded yet",
                    lines=10  # Increase height to show more content
                )
                
                file_upload.upload(
                    on_file_upload,
                    inputs=[file_upload],
                    outputs=[upload_status]
                )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(share=True)

#Execute the pipeline with output parser
'''
input_data = {
    "question": "What challenges exist regarding AI adoption?"}
result = pipeline_with_parser.invoke(input_data)
print(result)
'''


