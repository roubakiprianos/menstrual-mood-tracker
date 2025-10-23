import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import psycopg2
from psycopg2 import sql
import datetime

# =======================================================
# 1. CONFIGURATION AND AUTHENTICATION SETUP
# =======================================================

# --- Load Authentication Configuration from config.yaml ---
try:
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    st.error("Configuration file (config.yaml) not found. Please create it.")
    st.stop()
except Exception as e:
    st.error(f"Error loading config.yaml: {e}")
    st.stop()

# --- Initialize Authenticator ---
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# --- Initialize Database Connection ---
@st.cache_resource
def init_db_connection():
    try:
        # Connect using the DATABASE_URL secret
        conn = psycopg2.connect(st.secrets["DATABASE_URL"])
        return conn
    except Exception as e:
        st.error("Database connection failed. Please check your 'DATABASE_URL' secret/environment variable.")
        st.error(f"Details: {e}")
        st.stop()

# --- Load NLP Pipeline (Hugging Face) ---
@st.cache_resource
def load_emotion_pipeline():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

# --- Core Functions ---

def analyze_emotion(text, emotion_analyzer):
    """Analyzes text using the HuggingFace model and returns the result."""
    results = emotion_analyzer(text)[0]
    top_emotion = max(results, key=lambda x: x['score'])
    return top_emotion['label'], top_emotion['score'], results

def create_table_if_not_exists(conn):
    """Ensures the required database table exists."""
    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS journal_entries (
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(50) NOT NULL,
        entry_date DATE NOT NULL,
        period_day INTEGER,
        summary TEXT,
        emotion_label VARCHAR(50),
        confidence_score NUMERIC,
        joy_score NUMERIC,
        sadness_score NUMERIC,
        anger_score NUMERIC,
        fear_score NUMERIC,
        surprise_score NUMERIC,
        disgust_score NUMERIC,
        neutral_score NUMERIC
    );
    """
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
        conn.commit()

def load_user_history(conn, user_id):
    """Loads all historical data for the logged-in user."""
    query = "SELECT * FROM journal_entries WHERE user_id = %s ORDER BY entry_date DESC;"
    try:
        # Rename columns to match the application's expected DataFrame columns
        df = pd.read_sql(query, conn, params=[user_id])
        df.columns = [
            'id', 'User ID', 'Date', 'Period Day', 'Summary', 'Emotion Label', 'Confidence Score',
            'Joy_Score', 'Sadness_Score', 'Anger_Score', 'Fear_Score', 'Surprise_Score', 'Disgust_Score', 'Neutral_Score'
        ]
        # Ensure correct data types for plotting
        df['Date'] = pd.to_datetime(df['Date'])
        return df.drop(columns=['id', 'User ID']) # Drop internal columns for app display
    except Exception as e:
        # If table doesn't exist yet or other load error, return an empty structure
        st.warning(f"No history found or error loading data. Start logging! ({e})")
        return pd.DataFrame(columns=[
            'Date', 'Period Day', 'Summary', 'Emotion Label', 'Confidence Score',
            'Joy_Score', 'Sadness_Score', 'Anger_Score', 'Fear_Score', 'Surprise_Score', 'Disgust_Score', 'Neutral_Score'
        ])

# =======================================================
# 2. LOGIN WIDGET AND STATUS CHECK
# =======================================================

name, authentication_status, username = authenticator.login('Login', 'main') 

# --- Stop execution if user is not authenticated ---
if authentication_status == False:
    st.error('Username/password is incorrect')
    st.stop() 

if authentication_status == None:
    st.warning('Please enter your username and password to proceed.')
    st.stop() 

# =======================================================
# 3. AUTHENTICATED APPLICATION START
# =======================================================

if authentication_status:
    # Initialize all resources once login is successful
    conn = init_db_connection()
    create_table_if_not_exists(conn)
    emotion_analyzer = load_emotion_pipeline()

    # --- UI Elements for Logged-in User ---
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.title(f"Welcome, {name}!")

    # Load data specific to the current user
    if 'history_df' not in st.session_state:
        st.session_state.history_df = load_user_history(conn, username)

    # --- App Structure ---
    st.set_page_config(layout="wide", page_title="Menstruation Mood & Symptom Tracker")
    st.title("Menstruation Mood & Symptom Tracker 🩸")
    st.markdown("Log your mood and symptoms **only during your menstrual period** to reveal patterns linked to the flow day.")

    with st.expander("❓ How the Emotion Analysis Works"):
        st.markdown("""
        The application uses a **Hugging Face** pre-trained language model to analyze your text and determine the probability (Confidence Score) of **seven core emotions**. The score ranges from **0.0** (no evidence) to **1.0** (absolute certainty).
        """)
    st.markdown("---") 

    # --- Input and Submission Form ---
    with st.form(key='period_form'):
        col_date, col_day = st.columns([1, 2])
        
        with col_date:
            entry_date = st.date_input("Date of Period Entry", value="today")

        with col_day:
            period_day = st.slider("Menstruation Day", min_value=1, max_value=7, value=1, step=1)
            st.caption(f"Log symptoms for Day {period_day} of your period.")

        user_summary = st.text_area(
            "Mood & Symptom Summary (2-3 sentences)",
            max_chars=300,
            height=150,
            placeholder="e.g., Severe cramps started today and my mood is very low. I feel easily frustrated and just want to rest."
        )
        submit_button = st.form_submit_button(label='Analyze and Log Day')

    # --- Analysis and Logging ---
    if submit_button and user_summary:
        
        emotion_label, confidence_score, results = analyze_emotion(user_summary, emotion_analyzer)
        log_date = pd.to_datetime(entry_date).strftime("%Y-%m-%d")
        
        # Prepare data for dictionary (required for insertion)
        entry_dict = {
            'User ID': username, # Used for DB insertion
            'Date': log_date,
            'Period Day': period_day,
            'Summary': user_summary,
            'Emotion Label': emotion_label,
            'Confidence Score': confidence_score,
        }
        for item in results:
            entry_dict[f"{item['label'].capitalize()}_Score"] = item['score']

        # --- SAVE DATA TO POSTGRESQL ---
        try:
            with conn.cursor() as cur:
                INSERT_SQL = """
                INSERT INTO journal_entries 
                (user_id, entry_date, period_day, summary, emotion_label, confidence_score, 
                 joy_score, sadness_score, anger_score, fear_score, surprise_score, disgust_score, neutral_score) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """
                
                data = (
                    username, 
                    log_date, 
                    period_day, 
                    user_summary, 
                    emotion_label, 
                    confidence_score,
                    entry_dict['Joy_Score'], entry_dict['Sadness_Score'], entry_dict['Anger_Score'], 
                    entry_dict['Fear_Score'], entry_dict['Surprise_Score'], entry_dict['Disgust_Score'], 
                    entry_dict['Neutral_Score']
                )
                
                cur.execute(INSERT_SQL, data)
                conn.commit()
            
            # After successful DB insert, reload the user's data to update the session state
            st.session_state.history_df = load_user_history(conn, username)

            # --- Immediate Feedback (Aesthetic Update) ---
            st.success(f"Log successful and data SAVED for {log_date} (Menstruation Day {period_day})!")
            
            col_emotion, col_confidence = st.columns(2)

            with col_emotion:
                st.metric(label="Dominant Emotion", value=f"✨ {emotion_label.upper()} ✨")
                
            with col_confidence:
                st.metric(label="Confidence Score", value=f"{confidence_score:.2f}")

        except Exception as e:
            st.error(f"Failed to save entry to database. Details: {e}")
            conn.rollback()
        
        st.markdown("---")

    # =======================================================
    # 4. HISTORY AND VISUALIZATION
    # =======================================================
    if not st.session_state.history_df.empty:
        
        plot_df = st.session_state.history_df.copy()

        # --- Line Chart: Confidence Trend ---
        st.header("Mood Trend Over Time (Confidence) 📈")
        
        fig = px.line(
            plot_df,
            x='Date',
            y='Confidence Score',  
            color='Emotion Label',
            title='Daily Emotion Confidence Trend',
            markers=True,
            line_shape='spline'
        )
        
        fig.update_layout(yaxis_range=[0, 1.1])
        st.plotly_chart(fig, use_container_width=True)

        # --- Grouped Bar Chart: Aggregation by Period Day ---
        st.markdown("---")
        st.header("Emotional Patterns by Menstruation Day 🗓️")
        
        emotion_cols = [
            'Joy_Score', 'Sadness_Score', 'Anger_Score', 
            'Fear_Score', 'Surprise_Score', 'Disgust_Score', 'Neutral_Score'
        ]
        
        # Check if necessary columns exist and we have enough data
        if all(col in st.session_state.history_df.columns for col in emotion_cols) and len(st.session_state.history_df) >= 2:
            
            plot_df['Period Day'] = plot_df['Period Day'].astype(int) 
            
            melted_df = plot_df.melt(
                id_vars=['Period Day'],
                value_vars=emotion_cols,
                var_name='Emotion',
                value_name='Average Confidence'
            )
            
            period_agg = melted_df.groupby(['Period Day', 'Emotion'])['Average Confidence'].mean().reset_index()
            
            fig_period = px.bar(
                period_agg,
                x='Period Day',
                y='Average Confidence',
                color='Emotion',
                barmode='group',
                title='Average Emotion Scores by Period Day',
                color_discrete_map={
                    'Joy_Score': 'gold', 'Sadness_Score': 'blue', 'Anger_Score': 'red',
                    'Fear_Score': 'purple', 'Surprise_Score': 'orange', 'Disgust_Score': 'green',
                    'Neutral_Score': 'gray'
                }
            )
            
            fig_period.update_layout(
                yaxis_range=[0, 0.7], 
                legend_title_text='Emotion Score',
                xaxis_title="Menstruation Day"
            )
            st.plotly_chart(fig_period, use_container_width=True)
        else:
            st.warning("Log at least two entries with different Menstruation Days to see pattern comparisons.")

        # --- Raw History Table ---
        st.subheader("Journal History")
        st.dataframe(st.session_state.history_df.iloc[::-1], use_container_width=True)

    # --- Footer Credit ---
    st.caption("Developed by Rouba 🌸")
