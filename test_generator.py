import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load the Hugging Face API Key from your .env file
load_dotenv()

# --- UI Setup ---
st.set_page_config(page_title="Auto-Test Case Generator", page_icon="📋", layout="wide")
st.title("📋 AI Test Case Generator")
st.markdown("Paste a Jira User Story or feature requirement below, and I will generate a structured test suite.")

# --- Load AI Model ---
@st.cache_resource 
def load_generator_pipeline():
    
    # Swapped to the free-tier friendly 7B parameter version
    # Note: We added task="text-generation" to prevent server routing confusion
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct", 
        huggingfacehub_api_token="hf_bringyourownkey",
        max_new_tokens=1000,  # Increased slightly to allow for longer test suites
        temperature=0.3
    )

    chat_llm = ChatHuggingFace(llm=llm)
    
    prompt_template = """
    You are an expert QA Automation Architect with deep knowledge of enterprise testing.
    Your task is to analyze the following User Story/Requirement and generate a comprehensive suite of manual test cases.
    
    You must include three distinct sections:
    1. Positive Test Scenarios
    2. Negative Test Scenarios
    3. Edge Case Scenarios

    For each test case, provide the following structure:
    * **Test Case ID:** (e.g., TC-POS-01)
    * **Scenario:** (A brief title)
    * **Test Steps:** (Numbered step-by-step actions)
    * **Expected Result:** (What should happen)

    User Story / Requirement: 
    {requirement}

    Generate the Test Suite below in clear Markdown format:
    """
    
    prompt = PromptTemplate.from_template(prompt_template)
    
    # LCEL Assembly Line
    generator_chain = (
        {"requirement": RunnablePassthrough()} 
        | prompt 
        | chat_llm 
        | StrOutputParser()
    )
    
    return generator_chain

# Initialize the generator system
chain = load_generator_pipeline()

# --- The Interface ---
st.subheader("Requirement Input")
user_story = st.text_area(
    "Paste your User Story here:", 
    height=150, 
    placeholder="As a registered user, I want to reset my password using my email address so that I can regain access to my account if I forget it."
)

if st.button("Generate Test Cases", type="primary"):
    if user_story:
        with st.spinner("Analyzing requirement and writing test cases..."):
            
            # Run the user story through our LCEL chain
            response = chain.invoke(user_story)
            
            # Print the AI's generated test suite
            st.success("Test Suite Generated Successfully!")
            st.markdown("### Generated Test Scenarios")
            st.markdown(response)
            
    else:
        st.warning("Please enter a User Story to generate test cases.")