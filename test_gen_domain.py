import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

# --- UI Setup ---
st.set_page_config(page_title="Auto-Test Case Generator", page_icon="📋", layout="wide")
st.title("📋 AI Test Case Generator")
st.markdown("Select a business domain and paste a User Story to generate a structured test suite.")


DOMAIN_CONTEXTS = {
    "Generic Web App": "Focus on standard web UI testing, basic validations, and standard positive/negative user flows.",
    
    "Lead to Order (L2O)": """
    Focus strictly on our custom Lead to Order (L2O) lifecycle. 
    You must understand that our system follows this STRICT chronological process:
    1. Account Creation
    2. Contact Creation
    3. Opportunity Creation
    4. Quote Generation
    5. Contract Generation
    6. Order Creation
    7. Order Submission (This MUST be executed from the Workbench)
    8. Order Activation (This can ONLY occur after completing the Orchestration Plan)
    
    When writing Test Steps, ensure you respect this sequence. If a User Story focuses on a later stage (e.g., 'Order Activation' or 'Order Submission'), 
    your test setup and prerequisite steps must explicitly state that all preceding steps in the 8-step sequence have been successfully completed. 
    Always validate the Orchestration Plan status prior to activation.
    """,
    
    "Procure to Pay (P2P)": """
    Focus strictly on the Procurement lifecycle.
    Ensure test cases cover: Purchase Requisition creation, Vendor Selection, PO Generation, 
    Goods Receipt processing, and Invoice Verification.
    """
}
# --- Load AI Model ---
@st.cache_resource 
def load_generator_pipeline():
    
    base_llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        huggingfacehub_api_token="hf_bringyourownkey", 
        max_new_tokens=1000,
        temperature=0.3
    )
    chat_llm = ChatHuggingFace(llm=base_llm)
    
    # Notice we added {journey_context} to the prompt template
    prompt_template = """
    You are an expert QA Automation Architect with deep knowledge of enterprise testing.
    Your task is to analyze the following User Story/Requirement and generate a comprehensive suite of manual test cases.
    
    Business Domain Rules:
    {journey_context}
    (Ensure all generated test cases strictly adhere to the business rules above).
    
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
    
    # Cleaner LCEL Chain for multiple inputs
    generator_chain = prompt | chat_llm | StrOutputParser()
    
    return generator_chain

chain = load_generator_pipeline()

# --- The Interface ---
st.subheader("Business Domain Configuration")

# 1. Add a Dropdown for the user to select the journey
selected_journey = st.selectbox(
    "Select the Business Journey for this requirement:", 
    options=list(DOMAIN_CONTEXTS.keys())
)

st.subheader("Requirement Input")
user_story = st.text_area(
    "Paste your User Story here:", 
    height=150, 
    placeholder="As a sales rep, I want to convert a qualified lead into an account so that I can generate a quote."
)

if st.button("Generate Test Cases", type="primary"):
    if user_story:
        with st.spinner(f"Applying '{selected_journey}' rules and generating test cases..."):
            
            # Fetch the specific rules for the selected journey
            current_context = DOMAIN_CONTEXTS[selected_journey]
            
            # Pass BOTH the user story and the journey rules into the AI
            response = chain.invoke({
                "requirement": user_story,
                "journey_context": current_context
            })
            
            st.success("Test Suite Generated Successfully!")
            st.markdown("### Generated Test Scenarios")
            st.markdown(response)
            
    else:
        st.warning("Please enter a User Story to generate test cases.")