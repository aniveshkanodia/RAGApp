from dotenv import load_dotenv

# Load environment variables from a .env file for langsmith tracing
load_dotenv()

import os
import json
from typing import List, Dict, Any

# Set tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, UnstructuredWordDocumentLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
import gradio as gr

# Docling imports
from langchain_docling import DoclingLoader 
from langchain_docling.loader import ExportType
# Global variables
embeddings = None
vectordb = None
retriever = None
llm = None
rag_chain = None
PROMPT = None
TOP_K = 4


def load_document(file_path: str) -> List[Document]:
    """Load a document based on its file extension.
    
    For PDF files, uses DoclingLoader to preserve structure.
    Falls back to alternative PDF loader if DoclingLoader fails.
    For other file types, uses standard loaders.
    
    Args:
        file_path: Path to the file to load
        
    Returns:
        List of Document objects from the file
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == ".pdf":
        try:
            # Try DoclingLoader first (preserves structure with semantic chunking)
            loader = DoclingLoader(
                file_path=file_path,
                export_type=ExportType.DOC_CHUNKS
            )
            documents = loader.load()
        except Exception as e:
            # Fallback to PyPDFLoader if DoclingLoader fails
            # Common reasons: missing page dimensions, corrupted PDF, non-standard format
            error_msg = str(e).lower()
            if "page-dimensions" in error_msg or "page dimension" in error_msg or "dimension" in error_msg:
                # Use PyPDFLoader as fallback for PDFs with missing page dimension metadata
                from langchain_community.document_loaders import PyPDFLoader
                import warnings
                warnings.warn(
                    f"DoclingLoader failed due to missing page dimensions. "
                    f"Falling back to PyPDFLoader. Error: {str(e)}"
                )
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            else:
                # Re-raise if it's a different error
                raise ValueError(
                    f"Failed to load PDF with DoclingLoader: {str(e)}. "
                    f"PDF may be corrupted or have unsupported format."
                ) from e
    elif file_ext == ".txt":
        loader = TextLoader(file_path)
        documents = loader.load()
    elif file_ext in [".docx", ".doc"]:
        loader = UnstructuredWordDocumentLoader(file_path)
        documents = loader.load()
    else:
        raise ValueError(
            f"Unsupported file type: {file_ext}. "
            f"Supported types: .pdf, .txt, .docx, .doc"
        )
    
    return documents


def process_documents_for_chunking(
    documents: List[Document], 
    file_path: str
) -> List[Document]:
    """Process documents for chunking based on the export type.
    
    For PDF files with DOC_CHUNKS mode, documents are already chunked by HybridChunker.
    For PDFs loaded with PyPDFLoader (fallback), use standard text splitting.
    For other document types, use standard text splitting.
    
    Args:
        documents: List of Document objects
        file_path: Path to the file (to determine processing mode)
        
    Returns:
        List of processed Document chunks
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # PDF files loaded with DoclingLoader are already chunked by HybridChunker
    # Check if documents have docling metadata to determine if they were chunked
    if file_ext == ".pdf":
        # Check if documents have docling metadata (indicates DoclingLoader was used)
        has_docling_metadata = any(
            "dl_meta" in doc.metadata for doc in documents
        )
        if has_docling_metadata:
            # Already chunked by DoclingLoader, return as-is
            return documents
        else:
            # Loaded with fallback PyPDFLoader, need to chunk
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len,
                add_start_index=True
            )
            chunks = text_splitter.split_documents(documents)
            return chunks
    
    # For non-PDF files, use standard text splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def init_rag():
    """Initialize the RAG pipeline components."""
    global embeddings, vectordb, retriever, llm, rag_chain, PROMPT
    
    if rag_chain is not None:
        return
    
    # Initialize embeddings
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    
    # Initialize vector database
    vectordb = Chroma(
        collection_name="documents",
        embedding_function=embeddings,
        persist_directory="./db/chroma_db",
    )
    
    # Initialize retriever
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    
    # Initialize LLM
    llm = ChatOllama(model="qwen3:0.6b", keep_alive="2h", temperature=0)
    
    # Define prompt template (following langchain-docling example pattern)
    PROMPT = PromptTemplate.from_template(
        "Context information is below.\n"
        "---------------------\n"
        "{context}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, answer the query.\n"
        "Query: {input}\n"
        "Answer:\n"
    )
    
    # Build RAG pipeline using langchain_core.runnables
    # Step 1: Wrap document retrieval
    retrieve_docs = RunnableLambda(
        lambda input_dict: {
            "docs": retriever.invoke(input_dict["input"]),
            "question": input_dict["input"]
        }
    )
    
    # Step 2: Format prompt with context and question
    prompt_step = RunnableLambda(
        lambda input_dict: {
            "input": input_dict["question"],
            "context": "\n\n".join(doc.page_content for doc in input_dict["docs"]),
            "docs": input_dict["docs"]
        }
    )
    
    # Step 3: Get LLM response and format final response
    def call_llm_and_format(input_dict):
        """Call LLM and return structured response with answer and context."""
        formatted_prompt = PROMPT.format(
            context=input_dict["context"],
            input=input_dict["input"]
        )
        llm_response = llm.invoke(formatted_prompt)
        answer = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
        return {
            "answer": answer,
            "context": input_dict["docs"],
            "input": input_dict["input"]
        }
    
    llm_step = RunnableLambda(call_llm_and_format)
    
    # Build the complete pipeline
    rag_chain = retrieve_docs | prompt_step | llm_step
    

@traceable(name="RAGApp_v2")
def run_pipeline(question: str) -> Dict[str, Any]:
    """Run the RAG pipeline and return structured response with sources.
    
    Args:
        question: User question
        
    Returns:
        Dictionary with 'answer', 'context', and 'input' keys
    """
    global rag_chain
    
    init_rag()
    
    if rag_chain is None:
        raise RuntimeError("RAG chain not initialized")
    
    # Invoke the chain (following langchain-docling example pattern)
    response = rag_chain.invoke({"input": question})
    return response


def answer_question(question: str) -> str:
    """Answer a question using the RAG pipeline with source attribution.
    
    Args:
        question: User question
        
    Returns:
        Answer string with source information
    """
    response = run_pipeline(question)
    answer = response["answer"]
    
    # Always include sources
    sources_info = []
    for i, doc in enumerate(response.get("context", []), 1):
        source_info = f"\n\nSource {i}:"
        
        metadata = doc.metadata
        if metadata:
            # Deserialize JSON metadata if present (we stored dl_meta as JSON string)
            dl_meta = {}
            if "dl_meta" in metadata:
                dl_meta_value = metadata["dl_meta"]
                if isinstance(dl_meta_value, str):
                    try:
                        dl_meta = json.loads(dl_meta_value)
                    except (json.JSONDecodeError, TypeError):
                        dl_meta = {}
                elif isinstance(dl_meta_value, dict):
                    dl_meta = dl_meta_value
            
            # Add headings if available (from docling metadata)
            if "headings" in dl_meta:
                headings = dl_meta["headings"]
                if headings:
                    source_info += f"\n  Section: {', '.join(headings)}"
            
            # Add page number if available (simplified extraction)
            if "doc_items" in dl_meta:
                for item in dl_meta["doc_items"]:
                    provs = item.get("prov", [])
                    if provs and "page_no" in provs[0]:
                        source_info += f"\n  Page: {provs[0]['page_no']}"
                        break
            
            # Add source file/path
            if "source" in metadata:
                source_info += f"\n  Source: {metadata['source']}"
        
        # Add preview of content
        preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        source_info += f"\n  Preview: {preview}"
        
        sources_info.append(source_info)
    
    if sources_info:
        answer += "\n\n--- Sources ---" + "".join(sources_info)
    
    return answer


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
        
        # Load document (PDFs use DoclingLoader automatically)
        documents = load_document(file_path)
        
        # Process documents for chunking
        chunks = process_documents_for_chunking(documents, file_path)
        
        # Add metadata about the source file if not present
        # Note: ChromaDB only accepts simple types (str, int, float, bool, None)
        # So we need to serialize complex metadata like dl_meta to JSON strings
        for chunk in chunks:
            # Serialize complex metadata to JSON strings for ChromaDB compatibility
            cleaned_metadata = {}
            for key, value in chunk.metadata.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    # Simple types can be stored directly
                    cleaned_metadata[key] = value
                elif isinstance(value, (dict, list)):
                    # Complex types (like dl_meta) need to be serialized to JSON string
                    cleaned_metadata[key] = json.dumps(value) if value else None
                else:
                    # Convert other types to string
                    cleaned_metadata[key] = str(value) if value else None
            
            # Ensure source and filename are present
            if "source" not in cleaned_metadata:
                cleaned_metadata["source"] = file_path
            if "filename" not in cleaned_metadata:
                cleaned_metadata["filename"] = os.path.basename(file_path)
            
            # Update chunk metadata with cleaned version
            chunk.metadata = cleaned_metadata
        
        # Add to existing vector database
        vectordb.add_documents(documents=chunks)  # type: ignore[misc]
        
        # Provide detailed status
        file_ext = os.path.splitext(file_path)[1].lower()
        status = f"Successfully uploaded and indexed {len(chunks)} chunks from {os.path.basename(file_path)}"
        if file_ext == ".pdf":
            status += "\n(Processed with DoclingLoader - structure preserved)"
        
        return status
    except Exception as e:
        return f"Error processing file: {str(e)}"


def build_ui():
    """Build and return the Gradio UI."""
    init_rag()
    
    def on_submit(user_msg, chat_history):
        """Handle user message submission."""
        if not user_msg or not str(user_msg).strip():
            return chat_history, chat_history, ""
        
        # Get answer with sources
        answer = answer_question(str(user_msg))
        chat_history = (chat_history or []) + [(str(user_msg), answer)]
        return chat_history, chat_history, ""
    
    def on_file_upload(file):
        """Handle file upload."""
        if file is None:
            return "No file uploaded."
        
        status_msg = "Processing..."
        yield status_msg
        
        # Get file path - Gradio File component returns a path string or file object
        file_path = file.name if hasattr(file, 'name') else str(file)
        
        # Process the uploaded file
        result = process_and_index_file(file_path)
        yield result
    
    with gr.Blocks(title="RAG Chat") as demo:
        gr.Markdown("# Internal ChatBot")
        gr.Markdown(
            "Chat with your knowledge base by uploading documents."
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                chat = gr.Chatbot(
                    height=500,
                    label="Chat",
                    show_copy_button=True
                )
                msg = gr.Textbox(
                    placeholder="Ask a question…",
                    label="Your question",
                    lines=2
                )
                state = gr.State([])
                
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    clear = gr.Button("Clear Chat")
                
                # Both submit button and Enter key submit the question
                submit_btn.click(on_submit, [msg, state], [chat, state, msg])
                msg.submit(on_submit, [msg, state], [chat, state, msg])
                clear.click(lambda: ([], [], ""), None, [chat, state, msg])
            
            with gr.Column(scale=1):
                gr.Markdown("### Upload Document")
                gr.Markdown(
                    "*PDF files* *TXT files* and *Word Documents* are accept file types by the application.\n"
                    "*Other file types* support will added soon."
                )
                file_upload = gr.File(
                    label="Upload PDF, TXT, or DOCX file",
                    file_types=[".pdf", ".txt", ".docx", ".doc"]
                )
                upload_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    placeholder="No file uploaded yet",
                    lines=10
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

