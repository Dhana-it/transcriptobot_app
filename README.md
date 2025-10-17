import streamlit as st
import tempfile
import os
from typing import Optional, Dict, Any
import re

# Set page config
st.set_page_config(
    page_title="Transcriptobot - Meeting Transcription & Analysis",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple fallback class that doesn't require heavy ML libraries
class SimpleTranscriptobot:
    def __init__(self):
        self.name = "Simple Transcriptobot"
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """Placeholder transcription - returns demo text."""
        return """This is a sample meeting transcript. The quarterly review meeting began at 9 AM with all department heads present. 
        
Sarah from Marketing reported a 15% increase in lead generation this quarter, primarily attributed to the new digital campaign launched in July. She recommended increasing the social media budget by $10,000 for the next quarter.

John from Sales presented the Q3 numbers showing revenue growth of 12% compared to the same period last year. He noted that the enterprise client segment performed particularly well, contributing 60% of total revenue.

The Product team, led by Mike, discussed the upcoming feature releases planned for Q4. The mobile app update is scheduled for October 15th, and the web platform enhancement will follow in November.

Action items were assigned: Sarah will prepare a detailed social media strategy by Friday, John needs to follow up with three potential enterprise clients by next week, and Mike should provide updated development timelines by Wednesday."""
    
    def summarize(self, text: str, max_length: int = 120, min_length: int = 30) -> str:
        """Simple extractive summarization."""
        sentences = text.split('. ')
        # Take first few important sentences
        important_sentences = []
        
        for sentence in sentences:
            # Look for sentences with numbers, percentages, or key business terms
            if any(keyword in sentence.lower() for keyword in ['increase', 'decrease', 'revenue', 'growth', '%', 'budget', 'quarter', 'meeting']):
                important_sentences.append(sentence.strip())
                if len(important_sentences) >= 3:
                    break
        
        if not important_sentences:
            important_sentences = sentences[:2]  # Fallback to first 2 sentences
        
        summary = '. '.join(important_sentences)
        if not summary.endswith('.'):
            summary += '.'
        
        return summary
    
    def extract_action_items(self, text: str, max_length: int = 256) -> str:
        """Extract action items using keyword matching."""
        action_keywords = [
            'will prepare', 'needs to', 'should', 'must', 'assigned', 'follow up',
            'action item', 'todo', 'task', 'deadline', 'by friday', 'by next week',
            'by wednesday', 'scheduled for', 'planned for'
        ]
        
        sentences = text.split('. ')
        action_items = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in action_keywords):
                # Clean up the sentence
                if sentence and not sentence.endswith('.'):
                    sentence += '.'
                action_items.append(sentence)
        
        if action_items:
            return "• " + "\n• ".join(action_items)
        else:
            return "No specific action items found in the transcript."
    
    def process_meeting(self, audio_path: str, language: Optional[str] = None) -> Dict[str, str]:
        """Process meeting and return results."""
        transcript = self.transcribe(audio_path, language)
        summary = self.summarize(transcript)
        action_items = self.extract_action_items(transcript)
        
        return {
            "transcript": transcript,
            "summary": summary,
            "action_items": action_items
        }

# Advanced class that tries to load ML libraries
class AdvancedTranscriptobot:
    def __init__(self, whisper_model_size="base"):
        self.ml_available = False
        self.transcriber = None
        self.summarizer = None
        
        try:
            # Try to import and load ML libraries
            import whisper
            from transformers import pipeline
            
            # Load Whisper
            self.transcriber = whisper.load_model(whisper_model_size)
            
            # Try to load summarization model
            try:
                self.summarizer = pipeline(
                    "summarization",
                    model="sshleifer/distilbart-cnn-6-6",  # Even smaller model
                    device=-1  # Force CPU usage
                )
            except Exception as e:
                st.warning(f"Could not load summarization model: {str(e)}")
                self.summarizer = None
            
            self.ml_available = True
            st.success("✅ Advanced ML models loaded successfully!")
            
        except ImportError as e:
            st.warning(f"ML libraries not available: {str(e)}. Using simple text processing.")
            self.ml_available = False
        except Exception as e:
            st.warning(f"Error loading ML models: {str(e)}. Using simple text processing.")
            self.ml_available = False
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """Transcribe audio using Whisper if available."""
        if self.ml_available and self.transcriber:
            try:
                result = self.transcriber.transcribe(audio_path, language=language)
                return result["text"]
            except Exception as e:
                st.error(f"Error transcribing audio: {str(e)}")
                return "Error: Could not transcribe audio file."
        else:
            return "Error: Whisper not available. Please upload a text file instead."
    
    def summarize(self, text: str, max_length: int = 120, min_length: int = 30) -> str:
        """Summarize text using ML model if available."""
        if self.ml_available and self.summarizer and len(text.split()) > min_length:
            try:
                summary = self.summarizer(
                    text, max_length=max_length, min_length=min_length, do_sample=False
                )
                return summary[0]['summary_text']
            except Exception as e:
                st.warning(f"Error in ML summarization: {str(e)}. Using simple method.")
        
        # Fallback to simple summarization
        return self._simple_summarize(text)
    
    def _simple_summarize(self, text: str) -> str:
        """Simple extractive summarization fallback."""
        sentences = text.split('. ')
        important_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in ['increase', 'decrease', 'revenue', 'growth', '%', 'budget', 'quarter', 'meeting']):
                important_sentences.append(sentence.strip())
                if len(important_sentences) >= 3:
                    break
        
        if not important_sentences:
            important_sentences = sentences[:2]
        
        summary = '. '.join(important_sentences)
        if not summary.endswith('.'):
            summary += '.'
        
        return summary
    
    def extract_action_items(self, text: str, max_length: int = 256) -> str:
        """Extract action items."""
        action_keywords = [
            'will prepare', 'needs to', 'should', 'must', 'assigned', 'follow up',
            'action item', 'todo', 'task', 'deadline', 'by friday', 'by next week',
            'by wednesday', 'scheduled for', 'planned for'
        ]
        
        sentences = text.split('. ')
        action_items = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in action_keywords):
                if sentence and not sentence.endswith('.'):
                    sentence += '.'
                action_items.append(sentence)
        
        if action_items:
            return "• " + "\n• ".join(action_items)
        else:
            return "No specific action items found in the transcript."
    
    def process_meeting(self, audio_path: str, language: Optional[str] = None) -> Dict[str, str]:
        """Process meeting and return results."""
        transcript = self.transcribe(audio_path, language)
        summary = self.summarize(transcript)
        action_items = self.extract_action_items(transcript)
        
        return {
            "transcript": transcript,
            "summary": summary,
            "action_items": action_items
        }

@st.cache_resource
def load_transcriptobot(use_advanced_mode: bool = False, whisper_model_size: str = "base"):
    """Load appropriate transcriptobot based on available libraries."""
    if use_advanced_mode:
        return AdvancedTranscriptobot(whisper_model_size)
    else:
        return SimpleTranscriptobot()

def main():
    # Header
    st.title("🎙️ Transcriptobot")
    st.markdown("**AI-powered meeting transcription, summarization, and action item extraction**")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Mode selection
        use_advanced_mode = st.checkbox(
            "🚀 Advanced Mode (Use ML models)",
            value=False,
            help="Try to use advanced ML models. If they fail, fallback to simple processing."
        )
        
        demo_mode = st.checkbox(
            "🧪 Demo Mode (Use sample data)",
            value=True,
            help="Use demo data instead of uploading real audio files"
        )
        
        if use_advanced_mode and not demo_mode:
            # Whisper model selection
            whisper_model_size = st.selectbox(
                "Whisper Model Size",
                options=["tiny", "base", "small"],
                index=0,  # Default to "tiny" for better compatibility
                help="Smaller models are more stable"
            )
            
            # Language selection
            language_options = {
                "Auto-detect": None,
                "English": "en",
                "Spanish": "es",
                "French": "fr",
                "German": "de"
            }
            
            selected_language = st.selectbox(
                "Language",
                options=list(language_options.keys()),
                index=0,
                help="Select the primary language of your audio"
            )
        else:
            whisper_model_size = "tiny"
            selected_language = "English"
        
        # Summary settings
        st.subheader("📝 Summary Settings")
        max_summary_length = st.slider(
            "Max Summary Length",
            min_value=50,
            max_value=300,
            value=120,
            step=10
        )
    
    # Load model
    try:
        with st.spinner("Loading transcription system..."):
            bot = load_transcriptobot(use_advanced_mode, whisper_model_size)
        
        if use_advanced_mode:
            st.info("ℹ️ Advanced mode enabled. If ML models fail, simple processing will be used.")
        else:
            st.info("ℹ️ Simple mode enabled. Using keyword-based processing.")
            
    except Exception as e:
        st.error(f"❌ Error loading system: {str(e)}")
        return
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if demo_mode:
            st.header("🧪 Demo Mode")
            st.info("Demo mode uses sample meeting data to demonstrate functionality.")
            
            if st.button("🚀 Analyze Sample Meeting", type="primary"):
                try:
                    with st.spinner("Processing sample data..."):
                        progress_bar = st.progress(0)
                        
                        progress_bar.progress(50)
                        st.text("📝 Analyzing transcript...")
                        
                        result = bot.process_meeting("sample.wav")
                        
                        progress_bar.progress(100)
                        st.success("✅ Analysis complete!")
                    
                    st.session_state.results = result
                    
                except Exception as e:
                    st.error(f"❌ Error processing sample: {str(e)}")
        
        else:
            st.header("📤 Upload Audio")
            
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=['mp3', 'wav', 'mp4', 'm4a', 'flac', 'ogg'],
                help="Supported formats: MP3, WAV, MP4, M4A, FLAC, OGG"
            )
            
            if uploaded_file is not None:
                st.info(f"**File:** {uploaded_file.name}\n**Size:** {uploaded_file.size / 1024 / 1024:.2f} MB")
                
                st.audio(uploaded_file, format='audio/wav')
                
                if st.button("🚀 Process Audio", type="primary"):
                    if not use_advanced_mode:
                        st.warning("⚠️ Simple mode cannot process audio files. Please enable Advanced Mode or use Demo Mode.")
                        return
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_audio_path = tmp_file.name
                    
                    try:
                        with st.spinner("Processing audio..."):
                            progress_bar = st.progress(0)
                            
                            progress_bar.progress(25)
                            st.text("🎵 Transcribing audio...")
                            
                            language_code = language_options.get(selected_language) if use_advanced_mode else None
                            result = bot.process_meeting(temp_audio_path, language=language_code)
                            
                            progress_bar.progress(75)
                            st.text("📝 Generating summary and action items...")
                            
                            progress_bar.progress(100)
                            st.success("✅ Processing complete!")
                        
                        st.session_state.results = result
                        
                    except Exception as e:
                        st.error(f"❌ Error processing audio: {str(e)}")
                    finally:
                        if os.path.exists(temp_audio_path):
                            os.unlink(temp_audio_path)
    
    with col2:
        st.header("📋 Results")
        
        if hasattr(st.session_state, 'results') and st.session_state.results:
            results = st.session_state.results
            
            tab1, tab2, tab3 = st.tabs(["📄 Transcript", "📝 Summary", "✅ Action Items"])
            
            with tab1:
                st.subheader("Full Transcript")
                st.text_area(
                    "Transcript",
                    value=results["transcript"],
                    height=400,
                    label_visibility="collapsed"
                )
                
                st.download_button(
                    label="💾 Download Transcript",
                    data=results["transcript"],
                    file_name="transcript.txt",
                    mime="text/plain"
                )
            
            with tab2:
                st.subheader("Meeting Summary")
                st.markdown(results["summary"])
                
                st.download_button(
                    label="💾 Download Summary",
                    data=results["summary"],
                    file_name="summary.txt",
                    mime="text/plain"
                )
            
            with tab3:
                st.subheader("Action Items")
                st.markdown(results["action_items"])
                
                st.download_button(
                    label="💾 Download Action Items",
                    data=results["action_items"],
                    file_name="action_items.txt",
                    mime="text/plain"
                )
            
            st.divider()
            
            combined_output = f"""MEETING TRANSCRIPT AND ANALYSIS
===============================

SUMMARY:
{results['summary']}

ACTION ITEMS:
{results['action_items']}

FULL TRANSCRIPT:
{results['transcript']}
"""
            
            st.download_button(
                label="📦 Download Complete Report",
                data=combined_output,
                file_name="meeting_report.txt",
                mime="text/plain",
                type="secondary"
            )
        
        else:
            if demo_mode:
                st.info("👆 Click 'Analyze Sample Meeting' to see demo results.")
            else:
                st.info("👆 Upload an audio file and click 'Process Audio' to see results.")
    
    # Footer
    st.divider()
    with st.expander("ℹ️ About Transcriptobot"):
        st.markdown("""
        **Transcriptobot** offers two modes of operation:
        
        **🧪 Demo Mode:**
        - Uses sample meeting data
        - Works without any ML dependencies
        - Perfect for testing the interface
        
        **🚀 Advanced Mode:**
        - Attempts to use OpenAI Whisper for transcription
        - Uses transformer models for better summarization
        - Falls back to simple processing if models fail
        
        **Features:**
        - Automatic transcript generation
        - Intelligent meeting summarization
        - Action item extraction
        - Multiple export formats
        
        **Note:** Advanced mode requires additional dependencies and may not work on all systems.
        """)

if __name__ == "__main__":
    main()
