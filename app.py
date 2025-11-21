import streamlit as st
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login as hf_login

st.set_page_config(page_title="HF Transformers Chat", layout="wide")

st.title("Hugging Face Transformers Chat (Streamlit)")

# Sidebar: model and auth
with st.sidebar:
    st.header("Model & Auth")
    model_name = st.text_input("Model (repo/model)", value="gpt2", help="Enter Hugging Face model id, e.g. gpt2 or EleutherAI/gpt-neo-125M")
    hf_token = st.text_input("Hugging Face token (optional)", type="password", help="Use this for private models or higher rate limits")
    max_new_tokens = st.slider("Max new tokens", min_value=32, max_value=1024, value=256, step=32)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.7, step=0.05)
    top_p = st.slider("Top-p (nucleus sampling)", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
    do_sample = st.checkbox("Do sampling (recommended for chat)", value=True)
    st.markdown("---")
    st.markdown("Notes:")
    st.markdown("- Large models may not load on CPU.")
    st.markdown("- For private models provide a HF token.")

# Provide HF token to huggingface-hub if provided
if hf_token:
    try:
        hf_login(token=hf_token, add_to_git_credential=False)
        os.environ["HF_TOKEN"] = hf_token
    except Exception as e:
        st.error(f"Failed to login with provided token: {e}")

# Caching model load
@st.cache_resource(show_spinner=False)
def load_textgen_pipeline(model_id: str):
    device = 0 if torch.cuda.is_available() else -1
    kwargs = {}
    # Try to use device_map if GPU is present (transformers will attempt to place on GPU)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer for '{model_id}': {e}")

    # Some models may not have pad_token and need special handling
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id)
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_id}': {e}")

    # Make sure tokenizer has pad_token for generation if not present (avoid errors)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text_gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    return text_gen

# Session state for chat messages and loaded model id
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: {"role": "user"|"assistant", "text": "..."}
if "loaded_model" not in st.session_state:
    st.session_state.loaded_model = None
if "textgen" not in st.session_state:
    st.session_state.textgen = None

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Chat")
    chat_box = st.container()
    # Render chat
    def render_chat():
        chat_box.empty()
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['text']}")
            else:
                st.markdown(f"**Model:** {msg['text']}")
    render_chat()

    user_input = st.text_input("Type your message and press Enter", key="user_input")

    if st.button("Send") or (user_input and st.session_state.get("last_submitted") != user_input):
        # Handle submission
        msg_text = user_input.strip()
        if not msg_text:
            st.warning("Please enter a message.")
        else:
            st.session_state.messages.append({"role": "user", "text": msg_text})
            st.session_state.last_submitted = msg_text

            # Load model if necessary
            if not model_name:
                st.error("Please provide a model name in the sidebar.")
            else:
                # If the requested model is different than loaded, load it
                if st.session_state.loaded_model != model_name:
                    with st.spinner(f"Loading model {model_name} ..."):
                        try:
                            st.session_state.textgen = load_textgen_pipeline(model_name)
                            st.session_state.loaded_model = model_name
                        except Exception as e:
                            st.error(f"Error loading model: {e}")
                            st.session_state.messages.append({"role": "assistant", "text": f"Failed to load model: {e}"})
                            render_chat()
                            st.stop()

                # Build prompt from conversation history (simple concatenation)
                # You can improve this to follow model-specific chat formatting
                conversation = ""
                for m in st.session_state.messages:
                    role_tag = "User" if m["role"] == "user" else "Assistant"
                    conversation += f"{role_tag}: {m['text']}\n"
                prompt = conversation + "Assistant:"

                # Generate
                with st.spinner("Generating response..."):
                    try:
                        outputs = st.session_state.textgen(
                            prompt,
                            max_new_tokens=max_new_tokens,
                            do_sample=do_sample,
                            temperature=temperature,
                            top_p=top_p,
                            return_full_text=False,
                        )
                        # Pipeline may return a list of dicts
                        if isinstance(outputs, list) and len(outputs) > 0:
                            gen_text = outputs[0].get("generated_text", "")
                        elif isinstance(outputs, dict):
                            gen_text = outputs.get("generated_text", "")
                        else:
                            gen_text = str(outputs)
                        # Post-process: remove repeated prompt if present
                        # If the model returns the full prompt+response, try to trim prompt prefix
                        if gen_text.startswith(prompt):
                            gen_text = gen_text[len(prompt):].strip()
                        st.session_state.messages.append({"role": "assistant", "text": gen_text})
                    except Exception as e:
                        st.error(f"Generation failed: {e}")
                        st.session_state.messages.append({"role": "assistant", "text": f"Error generating: {e}"})

            # Clear input
            st.session_state.user_input = ""
            render_chat()

with col2:
    st.subheader("Conversation")
    if st.session_state.messages:
        for i, m in enumerate(st.session_state.messages[::-1], 1):
            role = "You" if m["role"] == "user" else "Model"
            st.write(f"{role}: {m['text']}")
    else:
        st.info("No messages yet. Type a message and send to begin.")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.last_submitted = None
        st.experimental_rerun()

st.markdown("---")
st.caption("Built with transformers and Streamlit")
