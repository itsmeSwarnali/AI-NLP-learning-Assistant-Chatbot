import os
import base64
from groq import Groq


def encode_image_bytes(image_bytes: bytes) -> str:
    """
    Convert image bytes into base64 string.
    Groq vision API needs image as base64 data URL.
    """
    return base64.b64encode(image_bytes).decode("utf-8")


def analyze_image_with_groq(
    image_bytes: bytes,
    mime_type: str,
    user_question: str = ""
) -> str:
    """
    Analyze an uploaded image using a Groq vision-capable model.
    Returns a text explanation of the image.
    """

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        return "GROQ_API_KEY is missing. Please add it to your .env file or Streamlit Cloud secrets."

    client = Groq(api_key=api_key)

    base64_image = encode_image_bytes(image_bytes)

    if not user_question.strip():
        user_question = "Explain what is inside this image in simple language."

    prompt = f"""
        You are a helpful multimodal AI/NLP learning assistant.

        Analyze the uploaded image carefully.
        Explain what is visible in the image.
        If the image contains a diagram, code, screenshot, lecture note, architecture, graph, or AI/NLP concept, explain it clearly and simply.

        User question:
        {user_question}
        """

    completion = client.chat.completions.create(
        model="qwen/qwen3.6-27b",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        temperature=0.2,
        max_completion_tokens=1024,
    )

    return completion.choices[0].message.content