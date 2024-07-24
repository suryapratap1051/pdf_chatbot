import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from pydantic import BaseModel
from typing import List
from io import BytesIO

app = FastAPI()

class Question(BaseModel):
    question: str

def get_pdf_text(pdf_docs):
    text = ""
    for pdf_doc in pdf_docs:
        # Convert the uploaded file to a BytesIO object
        pdf_file = BytesIO(pdf_doc.file.read())
        pdf_reader = PdfReader(pdf_file)

        # Extract text from each page
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    # Handle the list of uploaded files
    raw_text = get_pdf_text(files)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation_chain = get_conversation_chain(vectorstore)
    app.state.conversation_chain = conversation_chain
    return {"message": "Files processed successfully"}

@app.post("/ask/")
async def ask_question(question: Question):
    if not hasattr(app.state, 'conversation_chain'):
        return {"error": "No documents processed. Please upload PDF files first."}
    response = app.state.conversation_chain({'question': question.question})
    chat_history = response['chat_history']
    return {"response": [msg.content for msg in chat_history]}

@app.get("/", response_class=HTMLResponse)
async def main_page():
    html_content = """
    <html>
        <head>
            <title>Chat with multiple PDFs</title>
        </head>
        <body>
            <h1>Chat with multiple PDFs</h1>
            <form action="/upload/" enctype="multipart/form-data" method="post">
                <input name="files" type="file" multiple>
                <input type="submit" value="Process">
            </form>
            <form action="/ask/" method="post">
                <input name="question" type="text">
                <input type="submit" value="Ask">
            </form>
        </body>
    </html>
    """
    return html_content

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
