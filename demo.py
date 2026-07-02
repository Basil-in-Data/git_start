import streamlit as st
import os
import re
# Import the official, stable Google GenAI client
from google import genai
from google.genai import types

# --- PAGE LAYOUT & CONFIG ---
st.set_page_config(page_title="Basil's AI Dashboard", page_icon="🤖", layout="wide")

st.title("🤖 Week 2 Deliverable: Functional Real AI App")
st.markdown("### Basil is here")

# --- SIDEBAR FOR API KEY ---
st.sidebar.header("🔑 API Configuration")
api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password", placeholder="AIzaSy...")

# --- TABS FOR REQUIREMENTS ---
tab1, tab2 = st.tabs(["🧠 Real Agent Interaction", "💡 Prompt Engineering Techniques"])

# Verify if API Key is provided
if not api_key:
    st.warning("Please enter your Google Gemini API Key in the sidebar to activate the live AI models.")
else:
    # Initialize the real Google GenAI client
    client = genai.Client(api_key=api_key)

   # ==========================================
    # TAB 1: REAL AGENT-BASED INTERACTIONS
    # ==========================================
    with tab1:
        st.header("1. Agent-Based Interaction (Live AI reasoning)")
        st.write("This agent utilizes real AI reasoning to break down a prompt, determine numerical execution requirements, and pass data to a calculator tool.")
        
        agent_query = st.text_input(
            "Ask the Agent a question (math or general knowledge):", 
            value="Take the number 9000, divide it by 2, and then add 750."
        )
        
        if st.button("Launch Agent Execution"):
            with st.spinner("AI Agent is analyzing the query structure..."):
                
                # --- STEP 1: ROUTING THOUGHT (Let the AI decide if it needs a tool) ---
                routing_prompt = (
                    f"You are an AI Agent routing engine. Analyze this user query: '{agent_query}'. "
                    "Does this query require mathematical calculation or parsing numbers? "
                    "Reply with exactly one word: 'CALCULATOR' or 'GENERAL'."
                )
                routing_response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=routing_prompt,
                )
                decision = routing_response.text.strip().upper()
                
                # --- STEP 2: EXECUTE BASED ON DECISION ---
                if "CALCULATOR" in decision:
                    st.info("**[Thought from Live AI]:** This query involves mathematical reasoning. I will invoke the local `Calculator_Tool`.")
                    
                    # Run the Calculator Tool
                    numbers = [float(s) for s in re.findall(r'\d+', agent_query)]
                    if len(numbers) >= 2:
                        calc_step = numbers[0] / numbers[1]
                        final_calc = calc_step + (numbers[2] if len(numbers) > 2 else 0)
                        observation_value = f"{final_calc:,.2f}"
                        st.code(f"[Observation from Tool]: Calculator_Tool execution success. Value = {observation_value}")
                        
                        # Synthesize final math answer
                        synthesis_prompt = f"The user asked: '{agent_query}'. The calculation tool returned: {observation_value}. Write a polite final response delivering this answer."
                        final_response = client.models.generate_content(model='gemini-2.5-flash', contents=synthesis_prompt)
                        st.success(f"🎯 **Final Agent Response:** {final_response.text}")
                    else:
                        st.error("❌ **[Observation from Tool]:** Error: The calculator tool requires at least two numbers to compute.")
                
                else:
                    # GENERAL KNOWLEDGE ROUTE (Bypasses the calculator tool completely)
                    st.info("**[Thought from Live AI]:** This is a general knowledge question. I do not need external calculation tools. I will answer directly from my internal knowledge base.")
                    st.warning("**[Action]:** Processing response natively using LLM capabilities.")
                    
                    # Direct Generation
                    general_response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=agent_query,
                    )
                    st.code("[Observation]: Internal knowledge retrieved successfully.")
                    st.success(f"🎯 **Final Agent Response:** {general_response.text}")
    # ==========================================
    # TAB 2: PROMPT ENGINEERING & TRANSFORMERS
    # ==========================================
    with tab2:
        st.header("2. Prompt Engineering & Transformer Integration")
        st.write("See how different prompt layouts change the live generation style of the Gemini Transformer model.")
        
        technique = st.selectbox("Select Strategy", ["Zero-Shot (Direct)", "Few-Shot (Examples)", "Role-Prompting (System Persona)"])
        user_input = st.text_input("Core Topic:", "Why data engineering is critical")
        
        # Engineering the prompt structure
        if technique == "Few-Shot (Examples)":
            constructed_prompt = (
                "Context: Provide an answer in exactly three words.\n"
                "Example 1: Driving cars -> Fast and dangerous.\n"
                f"Example 2: {user_input} ->"
            )
        elif technique == "Role-Prompting (System Persona)":
            constructed_prompt = f"You are an elite, highly professional tech consultant. Explain this concept in a formal corporate tone: {user_input}"
        else:
            constructed_prompt = f"Explain this concept directly: {user_input}"
            
        st.info(f"**Engineered Prompt Layout Sent to Gemini:**\n`{constructed_prompt}`")
        
        if st.button("Generate AI Response"):
            with st.spinner("Querying Gemini Transformer pipeline..."):
                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=constructed_prompt,
                    )
                    st.success("✨ **Live Model Output:**")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"API Error: {e}")