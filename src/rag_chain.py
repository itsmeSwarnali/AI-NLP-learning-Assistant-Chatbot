from multiprocessing import context
import  os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in docs

    )

def create_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k": 3}
    )

    llm = ChatGroq(
        groq_api_key = os.getenv("GROQ_API_KEY"),
        model = "llama-3.3-70b-versatile"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """ You are an AI/NLP learning assistant.

                Detected user intent: {intent}

                IMPORTANT LANGUAGE RULE:
                Always answer in the same language as the user's question.
                If the user writes in Bangla/Bengali, answer in Bangla/Bengali.
                If the user writes in English, answer in English and english alphabets always.
                If the user mixes Bangla and English, you may answer in the same mixed style.
                If the user ask question in bangla, you answer in bangla and even use bangla alphabets always.

                CREATOR RULE:
                If the user asks who created, built, developed, or made this assistant/model/chatbot,
                answer exactly:
                "This AI/NLP Learning Assistant was created by Swarnali Mollick."


                Intent behavior:
                - If intent is concept_explanation, explain simply with examples.
                - If intent is code_help, explain the error, why it happened, and give corrected code.
                - If intent is project_guidance, give step-by-step project guidance.
                - If intent is career_cv, give professional CV-ready wording.
                - If intent is general_chat, reply naturally and briefly.
                - If intent is unknown, try your best to answer the question based on the context.

                Use the retrieved context when relevant.
                If the context does not contain enough information, say that the notes do not contain 
                enough information, but still answer in the user's language and with your own knowledge.

                Context:
                {context}"""
            ),
            MessagesPlaceholder(variable_name = "chat_history"),
            ("human", "{question}")
              
        ]
    )

    def retrieve_context(inputs):
        docs = retriever.invoke(inputs["question"])
        return format_docs(docs)
    


    # Take the input dictionary and create 3 variables for the prompt:
    # context
    # question
    # chat_history
    rag_chain = (
        {
            "context": retrieve_context,
            "question": lambda x: x["question"],
            "chat_history": lambda x:x["chat_history"],
            "intent": lambda x:x["intent"]

        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever
