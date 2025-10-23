
import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

# --- CACHED MODEL LOADING ---
@st.cache_resource
def load_emotion_pipeline():
    """Loads the HuggingFace emotion classification pipeline."""
    # This model classifies into: joy, sadness, anger, fear, surprise, disgust, neutral
    # We use st.cache_resource to load this large model only once.
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

# Initialize the pipeline
emotion_analyzer = load_emotion_pipeline()

# --- Core Emotion Analysis Function ---
def analyze_emotion(text):
    """Analyzes text using the HuggingFace model and returns the result."""
    # The pipeline returns a list of dictionaries, one for each emotion score
    results = emotion_analyzer(text)[0]
    
    # Find the top predicted emotion
    top_emotion = max(results, key=lambda x: x['score'])
    
    return top_emotion['label'], top_emotion['score'], results

# --- Streamlit App Logic ---
st.set_page_config(layout="wide", page_title="Menstruation Mood & Symptom Tracker")

st.title("Menstruation Mood & Symptom Tracker 🩸")
st.markdown("Log your mood and symptoms **only during your menstrual period** to reveal patterns linked to the flow day.")

# --- Score Explanation Expander (Updated for Emotion Labels) ---
with st.expander("❓ How the Emotion Analysis Works"):
    st.markdown("""
    The application uses a **Hugging Face** pre-trained language model to analyze your text and determine the probability (Confidence Score) of **seven core emotions**: Joy, Sadness, Anger, Fear, Surprise, Disgust, and Neutral.
    
    ### 📊 Confidence Score
    The **Confidence Score** ranges from **0.0** (no evidence) to **1.0** (absolute certainty) for each emotion. The app highlights the emotion with the highest score.
    """)
st.markdown("---") 

# --- 1. Initialize History (using session state) ---
if 'history_df' not in st.session_state:
    # Initialize DataFrame with columns for Date, Summary, and base analysis
    st.session_state.history_df = pd.DataFrame(columns=[
        'Date', 'Period Day', 'Summary', 'Emotion Label', 'Confidence Score'
    ])

# --- 2. Input and Submission (SIMPLIFIED PERIOD-FOCUSED FORM) ---
with st.form(key='period_form'): # Renamed key
    
    col_date, col_day = st.columns([1, 2])
    
    with col_date:
        # User must select the date for the entry
        entry_date = st.date_input("Date of Period Entry", value="today")

    with col_day:
        # User selects the day of the period flow (1-7)
        period_day = st.slider("Menstruation Day", 
                               min_value=1, max_value=7, value=1, step=1)
        st.caption(f"Log symptoms for Day {period_day} of your period.")

    # Input area for the user's summary
    user_summary = st.text_area(
        "Mood & Symptom Summary (2-3 sentences)",
        max_chars=300,
        height=150,
        placeholder="e.g., Severe cramps started today and my mood is very low. I feel easily frustrated and just want to rest."
    )
    submit_button = st.form_submit_button(label='Analyze and Log Day')


# --- 3. Analysis and Logging (UPDATED TO STORE ALL SCORES) ---
if submit_button and user_summary:
    
    # --- Emotion Analysis ---
    emotion_label, confidence_score, results = analyze_emotion(user_summary)
    
    # Format the date for logging
    log_date = pd.to_datetime(entry_date).strftime("%Y-%m-%d") # Use the user-selected date
    
    # Prepare the base entry dictionary
    entry_dict = {
        'Date': log_date,                           
        'Period Day': period_day,                  # <-- Logging the period day
        'Summary': user_summary,
        'Emotion Label': emotion_label,
        'Confidence Score': confidence_score
    }
    
    # Extract and add all 7 individual emotion scores to the dictionary
    for item in results:
        # Note the use of .capitalize() to match the expected column names (Joy_Score)
        entry_dict[f"{item['label'].capitalize()}_Score"] = item['score']
        
    # Create the new entry DataFrame
    new_entry = pd.DataFrame([entry_dict])
    
    # Append the new entry to the history
    st.session_state.history_df = pd.concat(
        [st.session_state.history_df, new_entry],
        ignore_index=True
    )

    # --- Immediate Feedback ---
    st.success(f"Log successful for {log_date} (Menstruation Day {period_day})!")
    
    st.markdown(f"""
    ### Analyzed Dominant Emotion: **{emotion_label.upper()}**
    **Confidence:** {confidence_score:.2f}
    """)
    
    st.markdown("---")


# --- 4. History and Visualization ---
if not st.session_state.history_df.empty:
    
    plot_df = st.session_state.history_df.copy()
    plot_df['Date'] = pd.to_datetime(plot_df['Date'])

    # --- Line Chart: Confidence Trend ---
    st.header("Mood Trend Over Time (Confidence)")
    
    # Plot the trend: Confidence Score colored by the predicted emotion
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
    st.header("Emotional Patterns by Menstruation Day 🩸")
    st.markdown("This chart compares the average confidence scores for all emotions across the different days of your menstrual flow.")
    
    emotion_cols = [
        'Joy_Score', 'Sadness_Score', 'Anger_Score', 
        'Fear_Score', 'Surprise_Score', 'Disgust_Score', 'Neutral_Score'
    ]
    
    # Check if necessary columns exist and we have enough data (at least two entries)
    if all(col in st.session_state.history_df.columns for col in emotion_cols) and len(st.session_state.history_df) >= 2:
        
        # 1. Melt the DataFrame to long format (Emotion vs. Score)
        plot_df['Period Day'] = plot_df['Period Day'].astype(int) 
        melted_df = plot_df.melt(
            id_vars=['Period Day'],
            value_vars=emotion_cols,
            var_name='Emotion',
            value_name='Average Confidence'
        )
        
        # 2. Group by both Period Day and Emotion, then calculate the mean
        period_agg = melted_df.groupby(['Period Day', 'Emotion'])['Average Confidence'].mean().reset_index()
        
        # 3. Create the Grouped Bar Chart
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
    # Display the newest entries first (reverse the DataFrame)
    st.dataframe(st.session_state.history_df.iloc[::-1], use_container_width=True)
    st.caption("Developed by Rouba 🌸")
