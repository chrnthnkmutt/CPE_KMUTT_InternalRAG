from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata
import os

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None 

    def __init__(self):
        os.environ["OPENAI_API_KEY"] = "ollama"
        
        self.model = ChatOpenAI(
            model="llama3.2:3b",
            temperature=0,
            base_url="http://localhost:11434/v1"
        )
        
        # Improved text splitter settings for better chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Reduced for more focused chunks
            chunk_overlap=150,  # Increased overlap for better context preservation
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Better separator hierarchy
        )
        
        # Enhanced prompt template for more accurate responses
        self.prompt = PromptTemplate.from_template(
            """
            You are a helpful assistant that answers questions based on the provided context. 
            Use ONLY the information from the context below to answer the question. 
            If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided context to answer this question."
            
            Context:
            {context}
            
            Question: {question}
            
            Answer: Based on the provided context,
            """
        )
    
    def clear(self):
        """Clear all stored documents and reset the assistant"""
        self.vector_store = None
        self.retriever = None
        self.chain = None
    
    def ingest(self, pdf_file_path: str):
        """Ingest PDF file and create vector store with improved processing"""
        try:
            # Load PDF
            loader = PyPDFLoader(pdf_file_path)
            docs = loader.load()
            
            # Enhanced document preprocessing
            for doc in docs:
                # Clean up text
                doc.page_content = doc.page_content.replace('\n\n', '\n').strip()
                # Add metadata for better retrieval
                if not hasattr(doc, 'metadata') or not doc.metadata:
                    doc.metadata = {}
                doc.metadata['source'] = pdf_file_path
                doc.metadata['page'] = doc.metadata.get('page', 0)
            
            # Split documents with improved settings
            chunks = self.text_splitter.split_documents(docs)
            chunks = filter_complex_metadata(chunks)
            
            # Add chunk metadata for better tracking
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = i
                chunk.metadata['chunk_size'] = len(chunk.page_content)
            
            # Create embeddings and vector store
            embeddings = FastEmbedEmbeddings(
                model_name="BAAI/bge-small-en-v1.5"  # Better embedding model
            )
            
            if self.vector_store is None:
                # Create new vector store
                self.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory="./chroma_db"  # Persist for reuse
                )
            else:
                # Add to existing vector store
                self.vector_store.add_documents(chunks)
            
            # Enhanced retriever with better search parameters
            self.retriever = self.vector_store.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance for diversity
                search_kwargs={
                    "k": 5,  # Retrieve more chunks for better context
                    "fetch_k": 10,  # Fetch more candidates
                    "lambda_mult": 0.7,  # Balance between relevance and diversity
                },
            )
            
            # Create enhanced chain with context formatting
            def format_docs(docs):
                """Format retrieved documents for better context"""
                formatted = []
                for i, doc in enumerate(docs):
                    page_info = f"[Page {doc.metadata.get('page', 'Unknown')}]"
                    formatted.append(f"{page_info} {doc.page_content}")
                return "\n\n".join(formatted)
            
            self.chain = (
                {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                | self.prompt
                | self.model
                | StrOutputParser()
            )
            
        except Exception as e:
            raise Exception(f"Error ingesting PDF: {str(e)}")
    
    def ask(self, query: str) -> str:
        """Ask a question with enhanced query processing"""
        if not self.chain:
            return "Please upload a PDF document first."
        
        try:
            # Enhanced query preprocessing
            processed_query = query.strip()
            if not processed_query:
                return "Please provide a valid question."
            
            # Get response
            response = self.chain.invoke(processed_query)
            
            # Optional: Get source information
            relevant_docs = self.retriever.invoke(processed_query)
            sources = set()
            for doc in relevant_docs:
                if 'page' in doc.metadata:
                    sources.add(f"Page {doc.metadata['page']}")
            
            # Add source information to response
            if sources:
                source_info = f"\n\n*Sources: {', '.join(sorted(sources))}*"
                response += source_info
            
            return response
            
        except Exception as e:
            return f"Error processing question: {str(e)}"
    
    def get_relevant_chunks(self, query: str, k: int = 3):
        """Get relevant document chunks for debugging/inspection"""
        if not self.retriever:
            return []
        
        try:
            docs = self.retriever.invoke(query)
            return [(doc.page_content[:200] + "...", doc.metadata) for doc in docs[:k]]
        except Exception as e:
            return [f"Error retrieving chunks: {str(e)}"]
