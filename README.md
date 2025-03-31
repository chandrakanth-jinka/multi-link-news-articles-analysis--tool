# Multi-URL Chat Assistant

A Streamlit web application that enables users to chat with content from multiple web links using Langchain, FAISS, and Hugging Face API integration.

![image](https://github.com/user-attachments/assets/26977697-0231-4614-99b7-44137377fe02)

![image](https://github.com/user-attachments/assets/5cfbb381-a2e8-4c4a-925d-c98d4e8cbea2)

![image](https://github.com/user-attachments/assets/3c27f6ee-2603-497b-b6d5-333e40743230)

## Features

- Ask questions about content from multiple web URLs
- Dynamic URL input fields
- Efficient content retrieval using FAISS vector search
- Powered by Qwen/QwQ-32B model from Hugging Face
- Export chat history functionality
- Clean, intuitive interface

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone this repository or download the files to your local machine.

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Application

1. Start the Streamlit app:
   ```
   streamlit run app.py
   ```

2. The application will open in your default web browser at `http://localhost:8501`.

## Usage

1. Enter the number of links you would like to analyze (1-10).
2. Input the URLs in the provided text fields.
3. Click the "Process URLs" button to load and process the content.
4. Once processing is complete, use the chat interface to ask questions about the content from the links.
5. Export your chat history using the "Export Chat History" button.

## Technical Details

- Uses RecursiveCharacterTextSplitter to split text into appropriate chunks
- Implements FAISS vector database for efficient similarity search
- Applies the map-reduce method for generating comprehensive responses
- Provides source attribution in answers

## Note on API Key

The application uses a Hugging Face API key for accessing the Qwen/QwQ-32B model. In a production environment, you should use environment variables or Streamlit secrets to store this key securely.

## Troubleshooting

- If you encounter issues with URL loading, check that the URLs are valid and accessible.
- Ensure you have a stable internet connection for API calls to Hugging Face.
- For large websites, processing may take some time - please be patient.

## License

This project is open source and available under the MIT License. 
