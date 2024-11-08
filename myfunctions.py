import os
import streamlit as st
import pyaudio
import wave
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.llms.base import LLM
from typing import Any, List, Optional, Dict
from groq import Groq
from langchain_community.callbacks.manager import get_openai_callback
from langchain.chains import RetrievalQA
from deepgram import (DeepgramClient, SpeakOptions)

class GroqLLMConfig(BaseModel):
    model_name: str = Field(..., description="The name of the Groq model to use.")
    temperature: float = Field(0.0, description="The temperature to use for sampling.")
    groq_api_key: str = Field(..., description="The API key for Groq.")

class GroqLLM(LLM):
    config: GroqLLMConfig
    client: Any = None

    def __init__(self, model_name: str, temperature: float = 0.0, groq_api_key: Optional[str] = None):
        super().__init__()
        groq_api_key = groq_api_key
        if not groq_api_key:
            raise ValueError("Groq API key must be provided or set as GROQ_API_KEY environment variable.")

        self.config = GroqLLMConfig(
            model_name=model_name,
            temperature=temperature,
            groq_api_key=groq_api_key
        )
        self.client = Groq(api_key=self.config.groq_api_key)

    @property
    def config(self) -> GroqLLMConfig:
        return self._config

    @config.setter
    def config(self, value: GroqLLMConfig):
        self._config = value

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.config.model_name,
            temperature=self.config.temperature,
        )
        return response.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "Groq"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.config.model_name, "temperature": self.config.temperature}
    
# Create a Groq LLM instance
llm = GroqLLM(
    model_name="llama3-8b-8192",  # "llama-3.1-70b-versatile"
    temperature=0.1,
    groq_api_key=os.getenv("GROQ_API_KEY")
)
# Define a custom prompt template
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. 

{context}

Question: {question}
Answer: """

PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"]
)

def setup_qa(vectorstore):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
    )
    return qa

def get_answer(vectorstore):
    # Retrieve the current chat history
    chat_history = st.session_state[st.session_state['current_chat']]

    # Format the chat history as a string to pass to the model
    history_str = ""
    for message in chat_history:
        if message["role"] == "user":
            history_str += f"User: {message['content']}\n"
        else:
            history_str += f"Assistant: {message['content']}\n"

    # Create the prompt by combining history with the current query
    full_prompt = f"{history_str}Assistant:"

    # Set up the QA chain
    qa = setup_qa(vectorstore)
    
    # Fetch the result with the entire conversation context
    with get_openai_callback() :
        result = qa({"query": full_prompt})  # Using the history as part of the query

    answer = result['result']
    source_documents = result['source_documents']
    
    response = (f"{answer}\n\n")

    # Initialize a set to keep track of unique source information
    unique_sources = set()

    for doc in source_documents:
        source_info = f"- {doc.metadata['source']}"
        if 'page' in doc.metadata:  # Only pdfs have pages.
            source_info += f", Page {doc.metadata['page']}."

        # Add the source info to the set if it's not already present
        if source_info not in unique_sources:
            unique_sources.add(source_info)

    # Now, add the unique sources to the response string
    if "I don't know" not in answer: 
        response += "Sources:\n"
        for source in unique_sources:
            response += f"{source}\n"

    return response

def gen_audio(text, audio_path):
    ### ~~ Cartesia ~~ ###
    #client = Cartesia(api_key=os.environ.get("CARTESIA_API_KEY"))
    #data = client.tts.bytes(
    #    model_id="sonic-english",
    #    transcript=text,
    #    voice_id="248be419-c632-4f23-adf1-5324ed7dbf1d",  # Professional Woman Voice
    #    # You can find the supported `output_format`s at https://docs.cartesia.ai/api-reference/tts/bytes
    #    output_format={
    #        "container": "mp3",
    #        "encoding": "pcm_f32le",
    #        "sample_rate": 44100,
    #    },
    #)
    #with open(audio_path, "wb") as f:
    #    f.write(data)

    ### ~~ Deepgram ~~ ###
    sourceless_text = text.split("Sources")[0].strip() # Remove sources from TTS 
    deepgram = DeepgramClient(os.environ.get("DEEPGRAM_API_KEY"))
    options = SpeakOptions(model='aura-asteria-en')
    deepgram.speak.v("1").save(audio_path, {"text":sourceless_text}, options)

def transcribe_audio(audio_file):
    client = Groq()
    with open(audio_file, "rb") as file:
        # Create a transcription of the audio file
        transcription = client.audio.transcriptions.create(
          file=(audio_file, file.read()), # Required audio file
          model="whisper-large-v3-turbo", # Required model to use for transcription
          prompt="Specify context or spelling",  # Change it to adapt to the context of the app
          response_format="json",  
          #language="en",  # Omitt to let Groq guess the language or you can specify one. 
          temperature=0.0  # The lower the temperature the more accurate the transcription is
        )
        return transcription.text

# Audio recording settings
FORMAT = pyaudio.paInt16  # Sets the audio format to a 16-bit signed integer
CHANNELS = 1  # Sets to mono channel recording, better for voice recognition
RATE = 44100  # Sample rate in Hz
CHUNK = 1024  # Chunk size, audio read in chunks of 1024 at a time
OUTPUT_FILE = "Audio/question.wav"  # Output file

# Initialize PyAudio
p = pyaudio.PyAudio()

# Start recording function
def start_recording():
    st.session_state['recording'] = True
    st.session_state['frames'] = []

    stream = p.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

    #with st.spinner("Recording audio..."):
    while st.session_state['recording']:
        data = stream.read(CHUNK)
        st.session_state['frames'].append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    
# Stop recording function
def stop_recording():
    st.session_state['recording'] = False
    
    # Save the recorded data as a WAV file
    wf = wave.open(OUTPUT_FILE, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(st.session_state['frames']))
    wf.close()

def get_audio_query():
    if os.path.exists("Audio/question.wav"):
        audio_query = transcribe_audio("Audio/question.wav")
        os.remove("Audio/question.wav") # Cleanup the audio file because i will have no use.
        return audio_query
    else:
        return None

def no_chat_msg():
    if st.session_state['lang']== 'en':
        st.markdown(
            """
            ### No chat selected
            Please select or create a new chat session from the sidebar.

            ### **Tips:**
            - Avoid changing the subject within the same chat. Instead, open a new chat for each new topic or question.
            """
        )
        return
    elif st.session_state['lang']=="fr":
        st.markdown(
            '''
            ### Aucun chat sélectionné
            Veuillez sélectionner ou créer une nouvelle session de chat dans la barre latérale.

            ### **Conseils :**
            - Évitez de changer de sujet dans le même chat. Ouvrez plutôt un nouveau chat pour chaque nouveau sujet ou question.
            '''
        )
        return
    
# Create a Groq LLM instance for chat naming
llm_chats = GroqLLM(
    model_name="llama3-8b-8192",  # "llama-3.1-70b-versatile"
    temperature=0.5,
    groq_api_key=os.getenv("GROQ_API_KEY")
)
# Define a custom prompt template for chat naming
template_chats = """Generate a single, easy-to-understand and short phrase with simple words that summarizes the provided text, while keeping key words, for a chat session name. Just return the phrase with no additional explanation.
If the given text has an acronym, don't try to explain it, keep the acronym with no explanation:
 
Text:{input_text}

Summary:"""

PROMPT_chats= PromptTemplate(
    template=template_chats, input_variables=["input_text"]
)

def gen_chat_name(input_text: str) -> str:
    prompt = PROMPT_chats.format(input_text=input_text)
    
    # Get the name from the LLM
    with get_openai_callback():
        summary = llm_chats(prompt)

    summary = summary.replace('"', '')
    return summary

def update_chat_name(query):
    """Update the current chat name based on the provided query, ensuring no name conflicts."""
    new_chat_name = gen_chat_name(query)

    # Check for existing names to avoid conflicts
    existing_names = {chat[1] for chat in st.session_state['chat_sessions']}
    
    # If the new name already exists, modify it
    if new_chat_name in existing_names:
        # Find a unique name by appending a number
        suffix = 1
        while f"{new_chat_name} ({suffix})" in existing_names:
            suffix += 1
        new_chat_name = f"{new_chat_name} ({suffix})"
    
    # Update the chat session
    for chat in st.session_state['chat_sessions']:
        if chat[0] == st.session_state['current_chat']:
            chat[1] = new_chat_name
            st.session_state['current_chat_name'] = new_chat_name
            break

def get_input_text():
    chat_input_msg = ""
    if st.session_state['lang']== 'en':
        chat_input_msg = "Your Message..."
    elif st.session_state['lang']=="fr":
        chat_input_msg = "Votre Message..."
    return chat_input_msg

def show_files_in_database(vectorstore):
    st.subheader("Sources in the DataBase")
    with st.expander("Click to expand"):
        # Collect all unique source filenames
        sources = set()
        for doc in vectorstore.docstore._dict.values():
            metadata = doc.metadata
            source = metadata.get("source", "Unknown")
            sources.add(source)  # Add to the set to avoid duplicates
        # Display the unique filenames
        for source in sources:
            st.write(source)

