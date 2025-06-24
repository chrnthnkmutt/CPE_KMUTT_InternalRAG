# ChatPDF - AI-Powered PDF Question Answering System

A Streamlit-based chatbot that allows you to upload PDF documents and ask questions about their content using local LLM integration.

## 🚀 Features

- **PDF Document Ingestion**: Upload and process PDF files for question answering
- **Local LLM Integration**: Uses Ollama with Llama 3.2 3B model for privacy-focused AI responses
- **Advanced RAG (Retrieval-Augmented Generation)**: Combines document retrieval with language generation
- **Interactive Chat Interface**: Clean, user-friendly chat interface with message history
- **Source Attribution**: Responses include page references from the source documents
- **Persistent Vector Storage**: Uses ChromaDB for efficient document storage and retrieval
- **Multiple Document Support**: Upload and query multiple PDFs simultaneously

## 🛠️ Technology Stack

- **Frontend**: Streamlit with streamlit-chat for UI
- **Backend**: LangChain for RAG pipeline
- **LLM**: Ollama (Llama 3.2 3B model)
- **Vector Database**: ChromaDB for document embeddings
- **Embeddings**: FastEmbed with BAAI/bge-small-en-v1.5 model
- **Document Processing**: PyPDF for PDF parsing

## 📋 Prerequisites

1. **Python 3.8+**
2. **Ollama installed and running locally**
   ```bash
   # Install Ollama (macOS/Linux)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull the required model
   ollama pull llama3.2:3b
   
   # Start Ollama server
   ollama serve
   ```

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd streamlit_chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Ollama is running**
   ```bash
   curl http://localhost:11434/api/tags
   ```

## 🚀 Usage

1. **Start the application**
   ```bash
   streamlit run main.py
   ```

2. **Access the web interface**
   - Open your browser and navigate to `http://localhost:8501`

3. **Upload PDF documents**
   - Use the file uploader to select one or more PDF files
   - Wait for the ingestion process to complete

4. **Ask questions**
   - Type your questions in the text input field
   - The AI will respond based on the content of your uploaded documents
   - Responses include source page references

## 🏗️ Architecture

### Core Components

- **main.py**: Streamlit application entry point and UI logic
- **rag.py**: RAG implementation with ChatPDF class
- **requirements.txt**: Python dependencies

### RAG Pipeline

1. **Document Ingestion**
   - PDF loading and text extraction
   - Text chunking with overlap for context preservation
   - Embedding generation using FastEmbed
   - Vector storage in ChromaDB

2. **Query Processing**
   - Question preprocessing
   - Semantic search using Maximum Marginal Relevance (MMR)
   - Context formatting and prompt engineering
   - LLM response generation with source attribution

### Key Features of the RAG Implementation

- **Smart Chunking**: Uses RecursiveCharacterTextSplitter with optimized chunk size (800 chars) and overlap (150 chars)
- **Advanced Retrieval**: MMR search for balanced relevance and diversity
- **Context Enhancement**: Includes page numbers and metadata in responses
- **Error Handling**: Comprehensive error handling throughout the pipeline

## 📁 Project Structure

```
streamlit_chatbot/
├── main.py              # Streamlit application
├── rag.py               # RAG implementation
├── requirements.txt     # Dependencies
├── chroma_db/          # Vector database (created after first use)
│   ├── chroma.sqlite3
│   └── [embedding files]
└── README.md           # This file
```

## ⚙️ Configuration

### Model Settings
- **LLM Model**: Llama 3.2 3B (via Ollama)
- **Embedding Model**: BAAI/bge-small-en-v1.5
- **Temperature**: 0 (deterministic responses)
- **Chunk Size**: 800 characters
- **Chunk Overlap**: 150 characters
- **Retrieved Chunks**: 5 per query

### Customization Options

You can modify the following in rag.py:

```python
# Model configuration
self.model = ChatOpenAI(
    model="llama3.2:3b",  # Change model
    temperature=0,        # Adjust creativity
    base_url="http://localhost:11434/v1"
)

# Chunking parameters
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # Adjust chunk size
    chunk_overlap=150,   # Adjust overlap
)

# Retrieval parameters
self.retriever = self.vector_store.as_retriever(
    search_kwargs={
        "k": 5,          # Number of chunks to retrieve
        "fetch_k": 10,   # Candidate pool size
    }
)
```

## 🔍 Troubleshooting

### Common Issues

1. **Ollama not running**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # Start Ollama if not running
   ollama serve
   ```

2. **Model not found**
   ```bash
   # Pull the required model
   ollama pull llama3.2:3b
   ```

3. **Memory issues with large PDFs**
   - Reduce chunk size in rag.py
   - Process PDFs in smaller batches

4. **ChromaDB persistence issues**
   - Delete the chroma_db folder to reset
   - Ensure write permissions in the project directory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the RAG framework
- [Streamlit](https://streamlit.io/) for the web interface
- [Ollama](https://ollama.ai/) for local LLM hosting
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [FastEmbed](https://github.com/qdrant/fastembed) for embeddings

## 📊 Performance Notes

- **Processing Speed**: Depends on document size and system resources
- **Memory Usage**: Approximately 2-4GB RAM for typical usage
- **Storage**: ChromaDB requires ~10MB per 100 pages of text
- **Accuracy**: Optimized for factual questions about document content

## 🔮 Future Enhancements

- [ ] Support for additional document formats (DOCX, TXT)
- [ ] Multi-language support
- [ ] Advanced filtering and search options
- [ ] Export chat history
- [ ] Integration with cloud-based LLMs
- [ ] Document summarization features
- [ ] Batch processing capabilities