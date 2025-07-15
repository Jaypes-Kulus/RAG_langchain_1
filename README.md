# LangChain Documentation Helper

This project enables you to ingest documentation (PDFs or HTML) into a Pinecone vector store and query it using LangChain and Streamlit.

---

## Setup

1. **Clone the repository** and navigate to the project folder.

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   If you encounter missing packages, install them as needed:
   ```bash
   pip install streamlit langchain langchain-community langchain-openai langchain-pinecone python-dotenv
   ```

3. **Set up your environment variables:**
   - Create a `.env` file and add your OpenAI and Pinecone API keys. Use the following name convention
   - PINECONE_API_KEY = 
   - OPENAI_API_KEY =
---

## Usage

### 1. Ingest Documentation (`ingestion.py`)

- **Add your document locations:**
  - For PDFs:  
    Replace `"folder_path"` in the line  
    ```python
    pdf_folder = "folder_path"
    ```  
    with the path to your PDF folder, for example:  
    ```python
    pdf_folder = r"C:\path\to\your\pdfs"
    ```
  - For HTML docs:  
    Uncomment and set the correct path in the `ReadTheDocsLoader` section:
    ```python
    # loader = ReadTheDocsLoader(
    #     "file_path/ folder_path", # Path to documentation source files
    #     encoding="utf-8"
    # )
    ```

- **Run the ingestion script:**
   ```bash
   python ingestion.py
   ```
   This will load your documents, split them into chunks, and upload them to your Pinecone index.  
   **Note:** Ensure your Pinecone index dimension matches your embedding model (e.g., 1536 for `text-embedding-3-small`).

---

### 2. Query Documentation (`main.py`)

- **Start the Streamlit app:**
   ```bash
   streamlit run main.py
   ```
- Enter your question in the UI. The app will use LangChain to search your ingested documents and return answers with source references.

---

## Notes

- Update all file paths in the code to match your local document locations.
- If you add new document types or loaders, install any additional dependencies as needed.

---

## Troubleshooting

- **Dimension mismatch error:**  
  If you see a Pinecone error about vector dimensions, recreate your Pinecone index with the correct dimension (e.g., 1536).
- **Missing packages:**  
  Install any missing Python packages using `pip install <package-name>`.

---

**Enjoy querying your documents!**