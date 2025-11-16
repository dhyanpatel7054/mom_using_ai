import os
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from typing import TypedDict
import streamlit as st
from pathlib import Path

# Load env vars
load_dotenv()
os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Initialize models
llm = GoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    api_key=os.getenv("API_KEY"),
    temperature=0.7
)
whisper_model = WhisperModel("base", device="cpu")

# State definition
class MeetingState(TypedDict):
    input: str
    transcript: str
    output: str

# Transcription node
def transcribe_audio_node(state: MeetingState) -> MeetingState:
    try:
        segments, _ = whisper_model.transcribe(state["input"])
        state["transcript"] = " ".join(segment.text for segment in segments)
    except Exception:
        state["transcript"] = ""
    return state

# Summarization node
def summarize_text_node(state: MeetingState) -> MeetingState:
    if not state.get("transcript"):
        state["output"] = "No transcript available for summarization"
        return state

    prompt = PromptTemplate(
        input_variables=["transcript"],
        template="""Create professional meeting minutes from this transcript:\n{transcript}
        give me specific 
        Main Participants
        Point of discussion
        Action Items """
    )
    formatted_prompt = prompt.format(transcript=state["transcript"])
    try:
        state["output"] = llm.invoke(formatted_prompt)
    except Exception as e:
        state["output"] = f"Summarization failed: {str(e)}"
    return state

# Conditional edge
def continue_after_transcription(state: MeetingState) -> str:
    if state.get("transcript"):
        return "summarize"
    else:
        return "end"
# Graph
def create_meeting_minutes_graph() -> CompiledStateGraph:
    graph = StateGraph(MeetingState)
    graph.add_node("transcribe", transcribe_audio_node)
    graph.add_node("summarize", summarize_text_node)
    graph.add_edge(START, "transcribe")
    graph.add_conditional_edges("transcribe", continue_after_transcription, {"summarize": "summarize", "end": END})
    graph.add_edge("summarize", END)
    return graph.compile()

def process_meeting_audio(audio_file_path: str) -> dict:
    graph = create_meeting_minutes_graph()
    initial_state: MeetingState = {"input": audio_file_path, "transcript": "", "output": ""}
    return graph.invoke(initial_state)

# ----------------- STREAMLIT APP -----------------
st.set_page_config(page_title="Meeting Minutes Generator", page_icon="📝", layout="centered")
st.title("🎙️ Meeting Minutes Generator")
st.write("Upload an audio file to transcribe and generate professional meeting minutes.")

uploaded_file = st.file_uploader("Upload audio file", type=["mp3", "wav"])

if uploaded_file is not None:
    temp_path = Path("temp_audio.mp3")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.info("Processing your audio... ⏳")
    result = process_meeting_audio(str(temp_path))

    st.subheader("📝 Transcript")
    st.write(result.get("transcript", "No transcript found."))

    st.subheader("📊 Meeting Minutes")
    st.write(result.get("output", "No summary available."))

    temp_path.unlink(missing_ok=True)
