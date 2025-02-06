# 📚 DocuMind AI - Intelligent Document Assistant

An advanced document analysis system combining the power of DeepSeek language models with RAG (Retrieval Augmented Generation) for intelligent document parsing and question answering.

## 🌟 Key Features

- **PDF Document Processing**: Seamlessly upload and process PDF documents
- **Intelligent Question Answering**: Ask natural language questions about your documents
- **Vector-Based Search**: Uses embedded vector search for accurate information retrieval
- **Dark Mode UI**: Clean, modern interface optimized for readability
- **Real-time Processing**: Get instant responses to your queries

## 🛠️ Technology Stack

- **Frontend**: Streamlit with custom CSS styling
- **Language Model**: DeepSeek (1.5B and 3B parameter variants)
- **Document Processing**: PDFPlumber
- **Vector Storage**: LangChain's InMemoryVectorStore
- **Embeddings**: Ollama Embeddings
- **Framework**: LangChain for RAG implementation

## 🚀 Getting Started

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Run the Application**
```bash
streamlit run app.py
```

3. **Additional Requirements**
- Ollama must be running locally on port 11434
- DeepSeek model must be available in your Ollama installation

## 💡 Usage

1. Upload your PDF document using the file uploader
2. Wait for the document to be processed
3. Ask questions about your document in natural language
4. Receive AI-generated responses based on the document content

## 🔧 System Components

- `app.py`: Main application with chat interface
- `rag_deepseek.py`: RAG implementation with document processing
- `requirements.txt`: Project dependencies

## 🎨 Features

- Custom-styled dark mode interface
- Intelligent document chunking with overlap
- Context-aware response generation
- Real-time chat interface
- Model selection functionality

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
