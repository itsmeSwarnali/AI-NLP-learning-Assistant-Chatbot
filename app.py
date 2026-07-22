import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage

from src.ingestion import load_text_documents
from src.vectorstore import create_vectorstore
from src.rag_chain import create_rag_chain
from src.intent_classifier import IntentClassifier


# =========================
# Streamlit page setup
# =========================

st.set_page_config(
    page_title="AI/NLP Learning Assistant",
    page_icon="🤖",
    layout="centered"
)



st.title("🤖 Hi I am an AI/NLP Learning Assistant")
st.caption(
    "Learn More about Large Language Models, LangChain, RAG, Embeddings, Transformers, etc."
)


# =========================
# Load environment variables
# =========================

load_dotenv(override=True)

# For Streamlit Cloud secrets
# These values must be set BEFORE creating the LangChain chain/model.
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


# Backward compatibility for older LangChain tutorials/versions
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv(
        "LANGSMITH_PROJECT",
        "AI-NLP-Learning-Assistant"
    )

# =========================
# Hugging Face model repo for DistilBERT intent classifier
# =========================

try:
    if "INTENT_MODEL_REPO" in st.secrets:
        os.environ["INTENT_MODEL_REPO"] = st.secrets["INTENT_MODEL_REPO"]
except Exception:
    # This avoids errors when running locally without Streamlit secrets.
    pass

# =========================
# Load RAG chain + classifier
# =========================

@st.cache_resource
def load_chain():
    """
    Do not rebuild this expensive object every time the app refreshes.
    Build it once and reuse it.
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
    # LangChain memory format: HumanMessage + AIMessage
    st.session_state.chat_history = []


# =========================
# Display old messages
# =========================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =========================
# User input
# =========================

user_input = st.chat_input("Ask me about AI/NLP...")


# =========================
# Main chat logic
# =========================

if user_input:
    # Save user message into UI history
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            creator_keywords = [
                "who created",
                "who made",
                "who built",
                "who developed",
                "creator",
                "developer",
                "কে তৈরি",
                "কে বানিয়েছে",
                "কে বানিয়েছে"
            ]

            # Direct answer for creator-related questions
            if any(keyword in user_input.lower() for keyword in creator_keywords):
                if any(bangla_word in user_input for bangla_word in ["কে", "তৈরি", "বানিয়েছে", "বানিয়েছে"]):
                    answer = "এই AI/NLP Learning Assistant তৈরি করেছেন Swarnali Mollick."
                else:
                    answer = "This AI/NLP Learning Assistant was created by Swarnali Mollick."

            else:
                # DistilBERT intent prediction
                intent, confidence = intent_classifier.predict(user_input)

                # Optional fallback if classifier confidence is low
                if confidence < 0.55:
                    intent = "concept_explanation"

                st.caption(f"Detected intent: {intent} | Confidence: {confidence:.2f}")

                # Run RAG chain
                answer = rag_chain.invoke(
                    {
                        "question": user_input,
                        "chat_history": st.session_state.chat_history,
                        "intent": intent
                    },
                    config={
                        "run_name": "AI_NLP_RAG_Chatbot_Response",
                        "tags": ["streamlit", "rag", "intent-classification"]
                    }
                )

            st.markdown(answer)

    # Save assistant reply into Streamlit UI history
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    # Save conversation in LangChain message format
    st.session_state.chat_history.extend(
        [
            HumanMessage(content=user_input),
            AIMessage(content=answer)
        ]
    )
