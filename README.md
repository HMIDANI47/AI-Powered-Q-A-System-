# AI-Powered-Q-A-System - with RAG
This project is a Retrieval-Augmented Generation (RAG) Chatbot that answers user questions based on a set of documents.
It uses Hugging Face models for text generation and FAISS for vector-based document retrieval.
The chatbot is built using Python and Flask, with a simple web interface for interact


**Features**
 - Document ingestion from both text files and PDFs
 - Text chunking with configurable size and overlap
 - Semantic search using FAISS vector database
 - Answer generation using Hugging Face's text generation models
 - Source attribution for transparency


**How It Works
The system works in four main steps:**

**Document Processing** : Text files and PDFs are read and chunked into smaller, manageable pieces while maintaining context through overlapping chunks.

**Embedding Generation** : Document chunks are converted into vector embeddings using a pre-trained Sentence Transformer model.
Vector Storage: Embeddings are stored in a FAISS index for efficient similarity search.

**Question Answering** : When a question is asked, the system:

- Embeds the question

-  Retrieves the most relevant document chunks

-  Constructs a prompt with the context

-  Generates an answer using the language model

**Customization**
You can customize the system by changing:

- The embedding model (any Sentence Transformers model)
- The language model (any Hugging Face text generation model)
- Chunk size and overlap parameters
- Number of retrieved chunks (top_k parameter)

**Limitations**

- The quality of answers depends on the quality and relevance of ingested documents
- Performance may degrade with very large document collections without additional optimization
- The system requires internet access for initial model downloads


**Installation :** 

1. Clone the Repository:
```
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
```

3. Create a Virtual Environment : `python -m venv myenv `

5. Activate the Virtual Environment:
   - On Windows:
       `myenv\Scripts\activate`
   - On macOS/Linux :
       `source myenv/bin/activate `
   
7. Install Dependencies:
      ` pip install -r requirements.txt

9. Running the Project  :
      `python app.py `


**requirements  :** 

Flask==3.0.0
nltk==3.8.1
PyPDF2==3.0.1
transformers==4.35.0
sentence-transformers==2.2.2
faiss-cpu==1.7.4
numpy==1.24.3
torch==2.1.0
