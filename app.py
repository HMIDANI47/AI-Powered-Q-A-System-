from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Tuple

# Document processing
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize

# Hugging Face modules
from transformers import AutoTokenizer, pipeline

# Vector database
import faiss
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

class DocumentProcessor:
    """Process and chunk documents from text files or PDFs."""
    
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Download NLTK data if not already available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def read_file(self, file_path: str) -> str:
        """Read content from a text file or PDF."""
        path = Path(file_path)
        
        if path.suffix.lower() == '.pdf':
            text = ""
            with open(path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        else:
            # Assume it's a text file
            with open(path, 'r', encoding='utf-8') as file:
                return file.read()
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Save the current chunk
                chunks.append(" ".join(current_chunk))
                
                # Keep some overlap for context continuity
                overlap_tokens = current_chunk[-self.chunk_overlap:] if self.chunk_overlap < len(current_chunk) else current_chunk
                current_chunk = overlap_tokens
                current_size = sum(len(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def process_documents(self, file_paths: List[str]) -> List[Tuple[str, str]]:
        """Process multiple documents and return chunks with their sources."""
        all_chunks = []
        
        for file_path in file_paths:
            text = self.read_file(file_path)
            chunks = self.chunk_text(text)
            file_name = os.path.basename(file_path)
            
            # Add source information to each chunk
            doc_chunks = [(chunk, file_name) for chunk in chunks]
            all_chunks.extend(doc_chunks)
        
        return all_chunks

class VectorStore:
    """Embeddings and vector database management using FAISS."""
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Load the embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Storage for document chunks
        self.documents = []
    
    def add_documents(self, doc_chunks: List[Tuple[str, str]]):
        """Add document chunks to the vector store."""
        texts = [chunk[0] for chunk in doc_chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts)
        
        # Add to FAISS index
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Store the document chunks
        self.documents.extend(doc_chunks)
        
        return len(doc_chunks)
    
    def search(self, query: str, k: int = 3) -> List[Tuple[str, str, float]]:
        """Search for relevant document chunks based on the query."""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search in the index
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        
        # Get the relevant chunks with their sources and distances
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents) and idx >= 0:  # Ensure index is valid
                chunk, source = self.documents[idx]
                distance = distances[0][i]
                results.append((chunk, source, float(distance)))
        
        return results

class RAGChatbot:
    """Retrieval-Augmented Generation chatbot with conversation history."""
    
    def __init__(self, 
                 llm_model_name: str = "google/flan-t5-base", 
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore(embedding_model_name)
        
        # Initialize the LLM
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm = pipeline("text2text-generation", model=llm_model_name, tokenizer=self.tokenizer)
        
        # Conversation history
        self.conversation_history = []
        self.max_history_length = 5  # Keep last 5 exchanges
    
    def ingest_documents(self, file_paths: List[str]) -> int:
        """Process and index documents."""
        doc_chunks = self.document_processor.process_documents(file_paths)
        num_chunks = self.vector_store.add_documents(doc_chunks)
        return num_chunks
    
    def generate_prompt(self, query: str, contexts: List[Tuple[str, str, float]]) -> str:
        """Generate a prompt for the LLM using retrieved contexts and conversation history."""
        context_text = "\n\n".join([f"Content from {src}: {txt}" for txt, src, _ in contexts])
        
        # Include recent conversation history
        history_text = ""
        if self.conversation_history:
            history_text = "Recent conversation:\n"
            for q, a in self.conversation_history:
                history_text += f"User: {q}\nAssistant: {a}\n"
        
        prompt = f"""
        Based on the following information, please answer the question.
        
        Context information:
        {context_text}
        
        {history_text}
        
        Current question: {query}
        
        Answer:
        """
        return prompt
    
    def answer_question(self, query: str, top_k: int = 3) -> Dict:
        """Answer a question using RAG approach with conversation context."""
        # Retrieve relevant contexts
        contexts = self.vector_store.search(query, k=top_k)
        
        # Generate prompt with contexts and history
        prompt = self.generate_prompt(query, contexts)
        
        # Generate answer with LLM
        response = self.llm(prompt, max_length=150, do_sample=False)[0]['generated_text']
        
        # Update conversation history
        self.conversation_history.append((query, response))
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
        
        # Return structured response
        return {
            "query": query,
            "answer": response,
            "sources": [{"content": txt, "source": src, "relevance": 1-dist} 
                        for txt, src, dist in contexts]
        }
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []
        return "Conversation history has been reset."

# Initialize chatbot
chatbot = RAGChatbot()

# Flask routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_files():
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
    
    files = request.files.getlist("files")
    file_paths = []
    for file in files:
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        file_paths.append(file_path)
    
    # Ingest documents
    num_chunks = chatbot.ingest_documents(file_paths)
    return jsonify({"message": f"Ingested {num_chunks} document chunks."})

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    query = data.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    # Get chatbot response
    result = chatbot.answer_question(query)
    return jsonify(result)

@app.route("/reset", methods=["POST"])
def reset_conversation():
    response = chatbot.reset_conversation()
    return jsonify({"message": response})

if __name__ == "__main__":
    # Create uploads directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    
    # Run Flask app with debug mode
    app.run(host="0.0.0.0", port=5000, debug=True)
