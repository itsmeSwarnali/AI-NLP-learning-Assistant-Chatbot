import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage

from src.ingestion import load_text_documents
from src.vectorstore import create_vectorstore
from src.rag_chain import create_rag_chain
from src.intent_classifier import IntentClassifier
from src.image_analyzer import analyze_image_with_groq


# =========================
# Streamlit page setup
# Must be the first Streamlit command
# =========================

st.set_page_config(
    page_title="MultiModal AI/NLP Learning Assistant",
    page_icon="🤖",
    layout="centered"
)


# =========================
# Load environment variables
# =========================

load_dotenv(override=True)

# For Streamlit Cloud secrets
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

    if "LANGSMITH_API_KEY" in st.secrets:
        os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
        os.environ["LANGSMITH_TRACING"] = st.secrets.get("LANGSMITH_TRACING", "true")
        os.environ["LANGSMITH_PROJECT"] = st.secrets.get(
            "LANGSMITH_PROJECT",
            "AI-NLP-Learning-Assistant"
        )

except Exception:
    # This avoids errors when running locally without Streamlit secrets.
    pass


# Backward compatibility for older LangChain versions
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv(
        "LANGSMITH_PROJECT",
        "AI-NLP-Learning-Assistant"
    )


# =========================
# App title
# =========================

st.title("🤖 Hi, I am a MultiModal AI/NLP Learning Assistant")
st.caption(
    "Ask about AI/NLP, upload diagrams, screenshots, lecture notes, or images, and I will explain them."
)


# =========================
# Load RAG chain + classifier
# =========================

@st.cache_resource
def load_chain():
    """
    Build expensive objects once and reuse them.
    """
    documents = load_text_documents("data/notes")
    vectorstore = create_vectorstore(documents)
    rag_chain, retriever = create_rag_chain(vectorstore)
    intent_classifier = IntentClassifier()

    return rag_chain, retriever, intent_classifier


rag_chain, retriever, intent_classifier = load_chain()


# =========================
# Session state
# =========================

if "messages" not in st.session_state:
    # UI display history
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    # LangChain memory format
    st.session_state.chat_history = []


# =========================
# Display old messages
# =========================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("content"):
            st.markdown(msg["content"])

        # Display previously uploaded images if stored
        for image in msg.get("images", []):
            st.image(
                image["bytes"],
                caption=image["name"],
                use_container_width=True
            )


# =========================
# Chat input with plus/attachment option
# =========================

chat_data = st.chat_input(
    "Ask me about AI/NLP or attach an image...",
    accept_file=True,
    file_type=["jpg", "jpeg", "png"]
)

if chat_data:
    user_input = chat_data.text or ""
    uploaded_files = chat_data.files or []
else:
    user_input = None
    uploaded_files = []


# =========================
# Main chat logic
# =========================

if user_input or uploaded_files:

    # If user only uploads image but writes nothing
    if not user_input:
        user_input = "Please explain this image."

    # Prepare image data for UI history
    uploaded_image_records = []

    for uploaded_file in uploaded_files:
        image_bytes = uploaded_file.getvalue()
        uploaded_image_records.append(
            {
                "name": uploaded_file.name,
                "type": uploaded_file.type,
                "bytes": image_bytes
            }
        )

    # Save user message into UI history
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
            "images": uploaded_image_records
        }
    )

    # Display current user message
    with st.chat_message("user"):
        st.markdown(user_input)

        for image in uploaded_image_records:
            st.image(
                image["bytes"],
                caption=image["name"],
                use_container_width=True
            )

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            creator_keywords = [
                "who created",
                "who create",
                "who made",
                "who built",
                "who developed",
                "who is your creator",
                "who is the creator",
                "created by",
                "made by",
                "built by",
                "developed by",
                "creator",
                "developer",
                "owner",
                "author",
                "কে তৈরি",
                "কে বানিয়েছে",
                "কে বানিয়েছে",
                "কে বানালো"
            ]

            user_text = user_input.lower().strip()

            # =========================
            # Direct creator answer
            # =========================

            if any(keyword in user_text for keyword in creator_keywords):
                if any(
                    bangla_word in user_input
                    for bangla_word in ["কে", "তৈরি", "বানিয়েছে", "বানিয়েছে", "বানালো"]
                ):
                    answer = "এই AI/NLP Learning Assistant তৈরি করেছেন Swarnali Mollick."
                else:
                    answer = "This AI/NLP Learning Assistant was created by Swarnali Mollick."

            else:
                # =========================
                # Image analysis
                # =========================

                image_analysis_texts = []

                if uploaded_image_records:
                    for image in uploaded_image_records:
                        with st.spinner(f"Analyzing image: {image['name']}"):
                            image_analysis = analyze_image_with_groq(
                                image_bytes=image["bytes"],
                                mime_type=image["type"],
                                user_question=user_input
                            )

                        image_analysis_texts.append(
                            f"Image name: {image['name']}\nImage analysis:\n{image_analysis}"
                        )

                    with st.expander("Image analysis result"):
                        for analysis in image_analysis_texts:
                            st.write(analysis)

                # =========================
                # Build final question for RAG
                # =========================

                if image_analysis_texts:
                    final_question = f"""
User question:
{user_input}

The user uploaded image(s). Below is the image analysis from a vision model:

{chr(10).join(image_analysis_texts)}

Now answer the user using:
1. the image analysis,
2. the user's question,
3. the retrieved AI/NLP notes if relevant.

Explain clearly and beginner-friendly.
"""
                else:
                    final_question = user_input

                # =========================
                # DistilBERT intent prediction
                # =========================

                # Intent should be predicted only from the user's original message
                intent, confidence = intent_classifier.predict(user_input)
                
                if confidence < 0.55:
                    intent = "concept_explanation"
                
                st.caption(f"Detected intent: {intent} | Confidence: {confidence:.2f}")
                
                answer = rag_chain.invoke(
                    {
                        "question": final_question,
                        "chat_history": st.session_state.chat_history,
                        "intent": intent
                    },
                    config={
                        "run_name": "AI_NLP_Multimodal_RAG_Response",
                        "tags": [
                            "streamlit",
                            "rag",
                            "intent-classification",
                            "image-input"
                        ]
                    }
                )

            st.markdown(answer)

    # Save assistant reply into Streamlit UI history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    # Save conversation in LangChain message format
    st.session_state.chat_history.extend(
        [
            HumanMessage(content=user_input),
            AIMessage(content=answer)
        ]
    )
