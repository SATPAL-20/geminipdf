import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as gen_ai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS  # Updated import statement
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import asyncio
from youtube_transcript_api import YouTubeTranscriptApi
import re

# Load environment variables
load_dotenv()

# Set up Google Gemini-Pro AI model
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
gen_ai.configure(api_key=GOOGLE_API_KEY)
model = gen_ai.GenerativeModel('gemini-pro')

# Function to translate roles between Gemini-Pro and Streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

# Function to clear chat history
def clear_chat_history():
    session_state.chat_session = model.start_chat(history=[])
    st.session_state.chat_session = session_state.chat_session
    st.session_state.messages = [
        {"role": "assistant", "content": "Chat history cleared. Ask me a question."}
    ]

# Function to read all PDF files and return text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# Function to get embeddings for each chunk
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to get conversational chain
async def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=gen_ai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input for PDF chat
async def pdf_user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    new_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = await get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )

    return response

# Function to summarize text
async def summarize_text(input_text):
    chain = await get_conversational_chain()

    summary_prompt = f"Summarize the following text:\n\n{input_text}\n\nSummary:"
    response = chain(
        {"input_documents": [], "question": summary_prompt}, return_only_outputs=True, )

    return response['output_text']

# Function to extract video ID from YouTube URL
def get_video_id(url):
    video_id = None
    # Regular expression patterns for different YouTube URL formats
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&]+)',  # https://www.youtube.com/watch?v=VIDEO_ID
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([^?]+)',  # https://youtu.be/VIDEO_ID
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^?]+)',  # https://www.youtube.com/embed/VIDEO_ID
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            break
    return video_id

# Function to get transcript from YouTube video
def get_youtube_transcript(video_url):
    try:
        video_id = get_video_id(video_url)
        if not video_id:
            st.error("Invalid YouTube URL")
            return None
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([d['text'] for d in transcript_list])
        return transcript
    except Exception as e:
        st.error(f"Error retrieving transcript: {e}")
        return None

# Initialize chat session
class SessionState:
    def __init__(self):
        self.chat_session = None

session_state = SessionState()

# Configure Streamlit page settings
st.set_page_config(
    page_title="Chat with EDUBOT!",
    page_icon=":brain:",  # Favicon emoji
    layout="wide",  # Page layout option
    initial_sidebar_state="expanded",
)

# CSS to hide Streamlit footer and customize styles
st.markdown("""
    <style>
        .css-18e3th9 {
            padding: 2rem 1rem;
        }
        .css-1d391kg {
            padding: 2rem 1rem;
        }
        .reportview-container .main footer {visibility: hidden;}
        .reportview-container .main .block-container {padding-top: 2rem;}
        .st-chat-message .css-1lhpgda {color: black;}
    </style>
""", unsafe_allow_html=True)

# Apply dark mode based on a condition (e.g., time of day, user preference stored in a database, etc.)
# For this example, we'll use a simple flag to toggle dark mode.
dark_mode = True  # Change this to False to disable dark mode

if dark_mode:
    st.markdown(
        """
        <style>
            .reportview-container {
                background: #1a1a1a;
                color: white;
            }
            .sidebar .sidebar-content {
                background: #1a1a1a;
                color: white;
            }
            .css-18e3th9, .css-1d391kg {
                background: #1a1a1a;
                color: white;
            }
            .st-chat-message .css-1lhpgda {color: white;}
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
            .reportview-container {
                background: white;
                color: black;
            }
            .sidebar .sidebar-content {
                background: white;
                color: black;
            }
            .css-18e3th9, .css-1d391kg {
                background: white;
                color: black;
            }
            .st-chat-message .css-1lhpgda {color: black;}
        </style>
        """,
        unsafe_allow_html=True
    )

# Main page
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Chat with PDF Files", "Chat with EDUBOT", "Text Summarizer", "YouTube Video Summarizer"))

    st.title("Chat with EDUBOT")

    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    if page == "Chat with PDF Files":
        st.title("Chat with PDF Files")
        with st.expander("Upload PDF Files"):
            pdf_docs = st.file_uploader(
                "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")

        st.write("Welcome to the chat!")

        if session_state.chat_session is None:
            session_state.chat_session = model.start_chat(history=[])
            st.session_state.chat_session = session_state.chat_session

        if "messages" not in st.session_state.keys():
            st.session_state.messages = [
                {"role": "assistant", "content": "Upload some PDFs and ask me a question"}]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input():
            st.session_state.messages.append(
                {"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = asyncio.run(pdf_user_input(prompt))
                    full_response = ''
                    for item in response['output_text']:
                        full_response += item
                    st.markdown(full_response)
            if response is not None:
                message = {"role": "assistant", "content": full_response}
                st.session_state.messages.append(message)

    elif page == "Text Summarizer":
        st.title("Text Summarizer")
        input_text = st.text_area("Enter the text you want to summarize:")

        if st.button("Summarize Text"):
            with st.spinner("Summarizing..."):
                summary = asyncio.run(summarize_text(input_text))
                st.write("Summary:")
                st.write(summary)

    elif page == "YouTube Video Summarizer":
        st.title("YouTube Video Summarizer")
        video_url = st.text_input("Enter YouTube Video URL:")

        if st.button("Get Transcript and Summarize"):
            with st.spinner("Retrieving transcript..."):
                transcript = get_youtube_transcript(video_url)
                if transcript:
                    with st.spinner("Summarizing transcript..."):
                        summary = asyncio.run(summarize_text(transcript))
                        st.write("Summary:")
                        st.write(summary)

    else:
        st.title("🤖 EDUBOT - ChatBot")
        # Display the chat history
        if session_state.chat_session is None:
            session_state.chat_session = model.start_chat(history=[])
            st.session_state.chat_session = session_state.chat_session

        for message in st.session_state.chat_session.history:
            with st.chat_message(translate_role_for_streamlit(message.role)):
                st.markdown(message.parts[0].text)

        # Input field for user's message
        user_prompt = st.chat_input("Ask EDUBOT...")
        if user_prompt:
            # Add user's message to chat and display it
            st.chat_message("user").markdown(user_prompt)

            # Send user's message to Gemini-Pro and get the response
            gemini_response = st.session_state.chat_session.send_message(
                user_prompt)

            # Display Gemini-Pro's response
            with st.chat_message("assistant"):
                st.markdown(gemini_response.text)

if __name__ == "__main__":
    main()
