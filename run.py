import os
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Union, Tuple

# Document processing
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize

# Hugging Face modules
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Vector database
import faiss
from sentence_transformers import SentenceTransformer

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
                chunks.append(" ".join(current_chunk))
                
                overlap_tokens = current_chunk[-self.chunk_overlap:] if self.chunk_overlap < len(current_chunk) else current_chunk
                current_chunk = overlap_tokens
                current_size = sum(len(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
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
            
            doc_chunks = [(chunk, file_name) for chunk in chunks]
            all_chunks.extend(doc_chunks)
        
        return all_chunks

class VectorStore:
    """Embeddings and vector database management using FAISS."""
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        self.index = faiss.IndexFlatL2(self.dimension)
        
        self.documents = []
    
    def add_documents(self, doc_chunks: List[Tuple[str, str]]):
        """Add document chunks to the vector store."""
        texts = [chunk[0] for chunk in doc_chunks]
        
        embeddings = self.embedding_model.encode(texts)
        
        self.index.add(np.array(embeddings).astype('float32'))
        
        self.documents.extend(doc_chunks)
        
        return len(doc_chunks)
    
    def search(self, query: str, k: int = 3) -> List[Tuple[str, str, float]]:
        """Search for relevant document chunks based on the query."""
        query_embedding = self.embedding_model.encode([query])
        
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents) and idx >= 0:  
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
        
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm = pipeline("text2text-generation", model=llm_model_name, tokenizer=self.tokenizer)
        
        self.conversation_history = []
        self.max_history_length = 5  
    
    def ingest_documents(self, file_paths: List[str]) -> int:
        """Process and index documents."""
        doc_chunks = self.document_processor.process_documents(file_paths)
        num_chunks = self.vector_store.add_documents(doc_chunks)
        return num_chunks
    
    def generate_prompt(self, query: str, contexts: List[Tuple[str, str, float]]) -> str:
        """Generate a prompt for the LLM using retrieved contexts and conversation history."""
        context_text = "\n\n".join([f"Content from {src}: {txt}" for txt, src, _ in contexts])
        
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
        contexts = self.vector_store.search(query, k=top_k)
        
        prompt = self.generate_prompt(query, contexts)
        
        response = self.llm(prompt, max_length=150, do_sample=False)[0]['generated_text']
        
        self.conversation_history.append((query, response))
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
        
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

def run_chatbot():
    """Run an interactive chatbot interface."""
    chatbot = RAGChatbot()
    
    print("=" * 50)
    print("RAG-Powered Chatbot System")
    print("=" * 50)
    print("\nPlease enter the paths to your documents (comma-separated):")
    doc_paths_input = input("> ")
    
    if doc_paths_input.strip():
        doc_paths = [path.strip() for path in doc_paths_input.split(",")]
        
        if not doc_paths or not any(os.path.exists(path) for path in doc_paths):
            print("\nNo valid documents found. Creating a sample document...")
            sample_doc = "sample_document.txt"
            with open(sample_doc, "w") as f:
                f.write("NEOV is an AI-powered company focused on automation and efficiency in the finance sector.")
            doc_paths = [sample_doc]
        
        print("\nIngesting documents...")
        num_chunks = chatbot.ingest_documents(doc_paths)
        print(f"Successfully ingested {num_chunks} document chunks.")
    else:
        print("\nNo documents provided. Creating a sample document...")
        sample_doc = "sample_document.txt"
        with open(sample_doc, "w") as f:
            f.write("NEOV is an AI-powered company focused on automation and efficiency in the finance sector.")
        
        num_chunks = chatbot.ingest_documents([sample_doc])
        print(f"Successfully ingested {num_chunks} document chunks from sample.")
    
    print("\n" + "=" * 50)
    print("Chat Interface - Type 'exit' to end or 'reset' to clear history")
    print("=" * 50)
    
    while True:
        print("\nYou: ", end="")
        query = input()
        
        if query.lower() in ['exit', 'quit', 'bye']:
            print("\nThank you for using the RAG chatbot. Goodbye!")
            break
        
        if query.lower() == 'reset':
            response = chatbot.reset_conversation()
            print(f"\nChatbot: {response}")
            continue
        
        try:
            result = chatbot.answer_question(query)
            
            # Display the answer
            print(f"\nChatbot: {result['answer']}")
            
            # Display sources 
            print("\nSources:")
            for i, source in enumerate(result['sources']):
                print(f"  {i+1}. {source['source']} (Relevance: {source['relevance']:.2f})")
        
        except Exception as e:
            print(f"\nChatbot: I'm sorry, I encountered an error: {str(e)}")

if __name__ == "__main__":
    run_chatbot()

