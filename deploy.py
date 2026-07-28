import streamlit as st
import pickle
import pandas as pd
import streamlit.components.v1 as components
import time

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="Enterprise ML Portal",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ADD THIS CSS BLOCK TO REMOVE TOP PADDING ---
st.markdown("""
    <style>
           .block-container {
                padding-top: 1.5rem; /* Reduces the top white space */
                padding-bottom: 1rem;
            }
    </style>
    """, unsafe_allow_html=True)

# App Header
st.title("Enterprise ML Portal")
st.markdown("A unified interface for model inference, business intelligence, and AI assistance.")

# 2. Create Navigation Tabs
tab_overview, tab_inference, tab_tableau, tab_chat = st.tabs([
    "📊 Project Overview", 
    "🧪 ML Inference", 
    "📈 Tableau Dashboard", 
    "🤖 GenAI Assistant"
])

# --- TAB 1: PROJECT OVERVIEW ---
with tab_overview:
    st.header("Project Overview")
    st.markdown("""
    Welcome to the model deployment portal. This tool bridges the gap between raw predictive analytics 
    and actionable business insights.
    
    ### Architecture Map
    *   **Tab 1:** High-level project metrics and documentation.
    *   **Tab 2:** Live model inference pipeline accepting custom feature inputs.
    *   **Tab 3:** Embedded BI reporting for historical trend analysis.
    *   **Tab 4:** Conversational AI layer for querying model logic and data.
    """)
    
    # Placeholder layout for high-level metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Accuracy", "94.5%", "+1.2%")
    col2.metric("Inference Latency", "124 ms", "-15 ms")
    col3.metric("Last Retrained", "24 Hours Ago", "Automated Pipeline")

# --- TAB 2: ML INFERENCE (PKL FILE) ---
with tab_inference:
    st.header("Model Testing & Inference")
    st.write("Input the required features below to generate a real-time prediction.")
    
    # Use a form so the app doesn't rerun on every single keystroke
    with st.form("prediction_form"):
        st.subheader("Feature Inputs")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            feature_1 = st.number_input("Feature 1 (Continuous)", value=0.0)
            feature_2 = st.number_input("Feature 2 (Continuous)", value=0.0)
        with col2:
            feature_3 = st.selectbox("Feature 3 (Categorical)", ["Category A", "Category B", "Category C"])
            feature_4 = st.slider("Feature 4 (Scaled)", 0, 100, 50)
        with col3:
            # Leave space or add more features
            pass
            
        submitted = st.form_submit_button("Generate Prediction", type="primary")
        
    if submitted:
        with st.spinner("Loading model and predicting..."):
            try:
                # --- ACTUAL IMPLEMENTATION ---
                # with open("model.pkl", "rb") as f:
                #     model = pickle.load(f)
                # prediction = model.predict([[feature_1, feature_2, feature_3, feature_4]])
                
                # --- MOCK IMPLEMENTATION FOR SKELETON ---
                time.sleep(1) # Simulate processing time
                mock_prediction = "Class 1 (High Probability)"
                
                st.success("Prediction generated successfully!")
                st.metric("Model Output", mock_prediction)
                
            except Exception as e:
                st.error(f"Error loading the model file. Ensure `model.pkl` is in the root directory. Error: {e}")

# --- TAB 3: TABLEAU DASHBOARD ---
with tab_tableau:
    st.header("Business Intelligence Dashboard")
    st.write("Live operational data synced directly from our Tableau Server.")
    
    # For Tableau, you need the public share link or server embed link.
    # The URL usually needs the specific embed parameters appended.
    tableau_embed_url = "https://public.tableau.com/views/Superstore_embedded_800x800/Overview?:showVizHome=no&:embed=true"
    
    # st.components.v1.iframe is highly reliable for external embeds
    components.iframe(tableau_embed_url, width=1000, height=850, scrolling=True)
    st.caption("Note: Ensure your Tableau dashboard permissions allow for iframe embedding.")

# --- TAB 4: GENAI CHATBOT ---
with tab_chat:
    st.header("GenAI Data Assistant")
    st.write("Ask questions about the model logic, feature engineering, or project documentation.")
    
    # 1. Initialize chat history in Streamlit session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am the project's AI assistant. How can I help you today?"}
        ]

    # 2. Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. Handle new user input
    if prompt := st.chat_input("Type your question here..."):
        
        # Add user message to state and display it immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # 4. Generate and display assistant response
        with st.chat_message("assistant"):
            # This is where you would call your LLM (OpenAI, LangChain, Vertex AI, etc.)
            # response = llm_chain.predict(input=prompt)
            
            mock_response = f"This is a placeholder response to your query: '{prompt}'. Connect your preferred LLM API here to generate real answers."
            st.markdown(mock_response)
            
        # Add assistant response to state so it persists on rerun
        st.session_state.messages.append({"role": "assistant", "content": mock_response})