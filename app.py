import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# 1. Load the Keys
load_dotenv()

# --- UI Setup ---
st.set_page_config(page_title="Tosca Log Analyzer", page_icon="🤖")
st.title("🤖 QA Log Analyzer (RAG)")
st.markdown("Ask me why a test suite failed, and I will analyze the execution logs!")

# --- Load Database & AI Models ---
@st.cache_resource 
def load_rag_pipeline():
    # Keep the Hugging Face embeddings for the database search
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # THE FIX: Enterprise-grade, perfectly stable Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1 
    )
    
    prompt_template = """
    You are an expert QA Automation Architect. Use the following execution logs to answer the tester's question. 
    If you don't know the answer based on the logs, just say you cannot find the error. Do not make up fake errors.

    Context Logs: {context}

    Question: {input}

    Helpful Answer:"""
    prompt = PromptTemplate.from_template(prompt_template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    answer_chain = prompt | llm | StrOutputParser()
    
    rag_chain = RunnableParallel(
        {"context": retriever, "input": RunnablePassthrough()}
    ).assign(answer=(
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"]))) 
        | answer_chain
    ))
    
    return rag_chain

# Initialize the RAG system
chain = load_rag_pipeline()

# --- The Chat Interface ---
user_query = st.text_input("Enter your question (e.g., 'Why did the E2E_Create_Purchase_Order list fail?'):")

if st.button("Analyze Logs"):
    if user_query:
        with st.spinner("Searching logs and analyzing..."):
            
            # Run the query through our custom LCEL chain
            response = chain.invoke(user_query)
            
            # Print the AI's final answer
            st.success("Analysis Complete!")
            st.write(response["answer"])
            
            # Show the source logs
            with st.expander("View Source Logs Found"):
                for i, doc in enumerate(response["context"]):
                    st.write(f"**Source {i+1}:** {doc.metadata.get('source', 'Unknown file')}")
                    st.text(doc.page_content)
    else:
        st.warning("Please enter a question first.")