import streamlit as st
import asyncio
import os
from inference.inference import Inference
import tempfile
import time
from io import BytesIO
import base64

st.set_page_config(
    page_title="Muyan-TTS Speech Generator",
    page_icon="🎙️",
    layout="wide",
)

# App title and description
st.title("🎙️ Muyan-TTS Speech Generator")
st.markdown("Generate speech from text using the Muyan-TTS model with voice cloning capabilities.")

# Create a container for the main content
main_container = st.container()

with main_container:
    # Sidebar for model selection
    with st.sidebar:
        st.header("Model Settings")
        model_type = st.selectbox("Model Type", ["base", "sft"], index=0)
        
        # Model paths
        if model_type == "base":
            model_path = "pretrained_models/Muyan-TTS"
        else:
            model_path = "pretrained_models/Muyan-TTS-SFT"
        
        cnhubert_model_path = "pretrained_models/chinese-hubert-base"
        
        st.markdown("---")
        st.markdown("### Model Information")
        st.markdown(f"**Selected Model**: Muyan-TTS {model_type.upper()}")
        st.markdown(f"**Model Path**: {model_path}")
        st.markdown(f"**CNHubert Path**: {cnhubert_model_path}")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Input")
        
        # Reference audio upload
        st.subheader("Reference Voice")
        reference_audio_option = st.radio(
            "Choose reference voice method",
            ["Upload Audio", "Use Default (Claire.wav)"]
        )
        
        ref_wav_path = "assets/Claire.wav"  # Default
        
        if reference_audio_option == "Upload Audio":
            uploaded_file = st.file_uploader("Upload reference audio (WAV format)", type=["wav"])
            if uploaded_file is not None:
                # Save the uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    ref_wav_path = tmp_file.name
                st.audio(uploaded_file, format="audio/wav")
        else:
            st.info("Using default 'Claire.wav' reference voice")
        
        # Text inputs
        st.subheader("Text Input")
        prompt_text = st.text_area(
            "Prompt Text (Optional context for voice cloning)", 
            "Although the campaign was not a complete success, it did provide Napoleon with valuable experience and prestige.",
            height=100
        )
        
        text_to_speak = st.text_area(
            "Text to Convert to Speech", 
            "Oh wow! [laughs] You will not believe what just happened! Yeah, what is up?",
            height=150
        )

    with col2:
        st.header("Output")
        
        # Audio output display
        st.subheader("Generated Speech")
        output_container = st.empty()
        
        # Generate button
        if st.button("Generate Speech", type="primary", use_container_width=True):
            with st.spinner("Setting up the model..."):
                try:
                    # Show a progress bar for model initialization
                    progress_bar = st.progress(0)
                    progress_bar.progress(10, "Initializing model...")
                    
                    async def generate_speech():
                        # Initialize model
                        tts = Inference(model_type, model_path, enable_vllm_acc=False)
                        progress_bar.progress(50, "Model initialized. Generating speech...")
                        
                        # Generate audio
                        wavs = await tts.generate(
                            ref_wav_path=ref_wav_path,
                            prompt_text=prompt_text,
                            text=text_to_speak
                        )
                        
                        # Get the first wav from the generator
                        wav_bytes = next(wavs)
                        progress_bar.progress(100, "Speech generated successfully!")
                        return wav_bytes
                    
                    # Run the async function
                    wav_bytes = asyncio.run(generate_speech())
                    
                    # Display the audio
                    output_container.audio(wav_bytes, format="audio/wav")
                    
                    # Provide download button
                    st.download_button(
                        label="Download Audio",
                        data=wav_bytes,
                        file_name="generated_speech.wav",
                        mime="audio/wav"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating speech: {str(e)}")
                    st.exception(e)
        
        # Additional information
        with st.expander("Advanced Options"):
            st.markdown("""
            ### Voice Cloning Process
            
            The model uses the reference audio to clone the voice characteristics 
            and applies them to the text you want to convert to speech.
            
            ### Tips for Better Results
            
            1. Use clear reference audio without background noise
            2. Keep the prompt text relevant to the voice style
            3. For emotional speech, use annotations like [laughs], [sighs], etc.
            """)

# Footer
st.markdown("---")
st.markdown("### About")
st.markdown(
    """
    This app uses Muyan-TTS, a text-to-speech model with voice cloning capabilities. 
    The model can mimic the voice characteristics from a reference audio sample.
    """
)
