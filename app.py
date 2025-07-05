import streamlit as st
import os
from PIL import Image
import numpy as np
from infer import inference, handle_inference_error
from chat import chatbot, getweatherdata, class_info_dict
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Weather-Based Pest Detection System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'detected_pests' not in st.session_state:
    st.session_state.detected_pests = []
if 'image_bytes' not in st.session_state:
    st.session_state.image_bytes = None
if 'language' not in st.session_state:
    st.session_state.language = "English"
if 'selected_day' not in st.session_state:
    st.session_state.selected_day = 0
if 'weather_data_cache' not in st.session_state:
    st.session_state.weather_data_cache = None

# Custom CSS
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 1rem;
        background-color: #1a1a1a;
    }
    
    /* White container for all content */
    .stApp > main > div > div > div {
        background-color: white;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
    }
    
    /* Title styling */
    h1 {
        color: #333;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    h3 {
        color: #333;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #0066cc;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: 500;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #0052a3;
    }
    
    /* File uploader styling */
    .stFileUploader {
        background-color: #f8f9fa;
        border: 2px dashed #dee2e6;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Chat container styling */
    .chat-container {
        background-color: #f5f5f5;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        height: 400px;
        overflow-y: auto;
    }
    
    /* Info box styling */
    .stAlert {
        background-color: #e3f2fd !important;
        color: #1976d2 !important;
        border-radius: 8px !important;
        border: 1px solid #bbdefb !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
        position: relative !important;
        z-index: 10 !important;
    }
    
    /* Column styling */
    [data-testid="column"] {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Dark background for main app */
    [data-testid="stAppViewContainer"] {
        background-color: #1a1a1a;
    }
    
    [data-testid="stHeader"] {
        background-color: #1a1a1a;
    }
    
    /* Image styling */
    .stImage {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Weather data horizontal layout */
    .weather-container {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.5rem;
    }
    
    .weather-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem;
        background-color: white;
        border-radius: 6px;
        flex: 1;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .weather-icon {
        font-size: 1.2rem;
    }
    
    .weather-info {
        display: flex;
        flex-direction: column;
    }
    
    .weather-label {
        font-size: 0.8rem;
        color: #666;
        font-weight: 500;
    }
    
    .weather-value {
        font-size: 1rem;
        color: #333;
        font-weight: 600;
    }
    
    /* Language selector styling */
    .stSelectbox {
        margin-bottom: 1rem;
    }
    
    /* Empty state message */
    .empty-state {
        text-align: center;
        padding: 2rem;
        background-color: #f5f5f5;
        border-radius: 10px;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to calculate GDD
def calculate_gdd(temp_max, temp_min, base_temp=10):
    """Calculate Growing Degree Days using the average method"""
    avg_temp = (temp_max + temp_min) / 2
    if avg_temp > base_temp:
        return avg_temp - base_temp
    return 0

# Helper function to get weather forecast with error handling
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_weather_forecast_with_gdd():
    days = ["Today", "Tomorrow", "2 days", "3 days", "4 days"]
    forecast_data = []
    accumulated_gdd = 0
    
    try:
        for day in days:
            weather = getweatherdata(day)
            
            # Check if weather data is None
            if weather is None:
                # Return dummy data if API fails
                return get_dummy_weather_data()
            
            daily_gdd = calculate_gdd(weather[6], weather[7])
            accumulated_gdd += daily_gdd
            
            forecast_data.append({
                'day': day,
                'date': weather[0],
                'temp': weather[1],
                'humidity': weather[2],
                'windspeed': weather[3],
                'description': weather[5],
                'temp_max': weather[6],
                'temp_min': weather[7],
                'daily_gdd': round(daily_gdd, 1),
                'accumulated_gdd': round(accumulated_gdd, 1)
            })
        return forecast_data
    except Exception as e:
        st.warning(f"Weather data unavailable. Using sample data. Error: {str(e)}")
        return get_dummy_weather_data()

# Dummy weather data for when API fails
def get_dummy_weather_data():
    """Return dummy weather data when API is unavailable"""
    base_date = datetime.now()
    days = ["Today", "Tomorrow", "2 days", "3 days", "4 days"]
    forecast_data = []
    accumulated_gdd = 0
    
    for i, day in enumerate(days):
        temp = 20 + i  # Sample temperature
        daily_gdd = calculate_gdd(temp + 5, temp - 5)
        accumulated_gdd += daily_gdd
        
        forecast_data.append({
            'day': day,
            'date': (base_date + timedelta(days=i)).strftime("%Y-%m-%d"),
            'temp': temp,
            'humidity': 65 + i * 2,
            'windspeed': 3.5 + i * 0.5,
            'description': 'Clear sky',
            'temp_max': temp + 5,
            'temp_min': temp - 5,
            'daily_gdd': round(daily_gdd, 1),
            'accumulated_gdd': round(accumulated_gdd, 1)
        })
    return forecast_data

# Function to create weather chart
def create_weather_chart(forecast_data, language):
    fig = go.Figure()
    
    # Get current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Add temperature line
    fig.add_trace(go.Scatter(
        x=[d['day'] for d in forecast_data],
        y=[d['temp'] for d in forecast_data],
        mode='lines+markers+text',
        name='Temperature',
        line=dict(color='#5470c6', width=3),
        marker=dict(
            size=15,
            color='#5470c6',
            line=dict(color='white', width=2)
        ),
        text=[f"{d['temp']}°C" for d in forecast_data],
        textposition="top center",
        textfont=dict(size=14, color='#333', weight='bold'),
        hovertemplate='<b>%{x}</b><br>Temperature: %{y}°C<extra></extra>',
        customdata=list(range(len(forecast_data)))
    ))
    
    # Add current date annotation
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"📅 {current_date}",
        showarrow=False,
        font=dict(size=12, color='#666'),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#ddd",
        borderwidth=1,
        borderpad=4
    )
    
    # Update layout
    fig.update_layout(
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        xaxis=dict(
            showgrid=True,
            gridcolor='#e0e0e0',
            gridwidth=1,
            zeroline=False,
            tickfont=dict(size=12, color='#333')
        ),
        yaxis=dict(
            title=dict(
                text='Temperature (°C)' if language == "English" else '気温 (°C)',
                font=dict(size=14, color='#333')
            ),
            showgrid=True,
            gridcolor='#e0e0e0',
            gridwidth=1,
            zeroline=False,
            range=[min([d['temp'] for d in forecast_data]) - 3, 
                   max([d['temp'] for d in forecast_data]) + 3],
            tickfont=dict(size=12, color='#333')
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial"
        )
    )
    
    return fig

# Title - ALWAYS IN ENGLISH
st.markdown("<h1>Pest Detection System</h1>", unsafe_allow_html=True)

# Create main container
with st.container():
    # Create two columns with spacing
    col1, spacer, col2 = st.columns([5, 0.5, 5])
    
    # Left column - Image upload and processing
    with col1:
        # Preview section
        preview_text = "🐛 プレビュー" if st.session_state.language == "Japanese" else "🐛 Preview"
        st.markdown(f"### {preview_text}")
        
        # File uploader
        upload_text = "ファイルを選択" if st.session_state.language == "Japanese" else "Choose file"
        help_text = "害虫を検出するために作物の画像をアップロードしてください" if st.session_state.language == "Japanese" else "Upload an image of crops to detect pests"
        
        uploaded_file = st.file_uploader(
            upload_text, 
            type=["jpg", "jpeg", "png"], 
            key="file_uploader",
            help=help_text
        )
        
        # Empty state message right below file uploader
        if not uploaded_file:
            if st.session_state.language == "Japanese":
                empty_message = "👆 害虫検出を開始するには、画像をアップロードしてください"
            else:
                empty_message = "👆 Please upload an image to start pest detection"
                
            st.markdown(
                f"""
                <div class="empty-state">
                    <p style='color: #666; font-size: 1rem; margin: 0;'>
                        {empty_message}
                    </p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        if uploaded_file is not None:
            # Read and store image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            st.session_state.image_bytes = image_np
            
            # Display uploaded image
            caption_text = "アップロードされた画像" if st.session_state.language == "Japanese" else "Uploaded Image"
            st.image(image, caption=caption_text, use_container_width=True)
            
            # Process button
            process_text = "処理" if st.session_state.language == "Japanese" else "Process"
            if st.button(process_text, type="primary", use_container_width=True):
                with st.spinner("処理中..." if st.session_state.language == "Japanese" else "Processing image..."):
                    # Perform pest detection
                    infer_img, classes, detected_classes = inference(st.session_state.image_bytes)
                    
                    # Store detected pests
                    st.session_state.detected_pests = []
                    for cls_idx in detected_classes:
                        pest_name = classes[int(cls_idx)]
                        if pest_name in class_info_dict:
                            st.session_state.detected_pests.append({
                                "pest": pest_name,
                                "info": class_info_dict[pest_name]
                            })
                    
                    st.session_state.processed = True
                    st.session_state.annotated_image = infer_img
        
        # Display processed image if available
        if st.session_state.processed and hasattr(st.session_state, 'annotated_image'):
            processed_text = "処理済み画像" if st.session_state.language == "Japanese" else "Processed Image"
            st.markdown(f"### {processed_text}")
            
            # Create caption with detected pests
            if st.session_state.detected_pests:
                pests_list = [p["pest"] for p in st.session_state.detected_pests]
                if st.session_state.language == "Japanese":
                    caption = f"**検出された害虫:** {', '.join(pests_list)}"
                else:
                    caption = f"**Detected Pests:** {', '.join(pests_list)}"
            else:
                caption = "**害虫は検出されませんでした**" if st.session_state.language == "Japanese" else "**No pests detected**"
            
            # Display annotated image
            st.image(st.session_state.annotated_image, caption=caption, use_container_width=True)
    
    # Spacer column
    with spacer:
        st.empty()
    
    # Right column - Pest Control Assistant and Weather
    with col2:
        # Pest Control Assistant header
        assistant_text = "💬 害虫対策アシスタント" if st.session_state.language == "Japanese" else "💬 Pest Control Assistant"
        st.markdown(f"### {assistant_text}")
        
        # Language selector - BELOW the assistant header
        def update_language():
            st.session_state.language = st.session_state.language_selector
        
        language = st.selectbox(
            "Select Language / 言語を選択",
            ["English", "Japanese"],
            index=0 if st.session_state.language == "English" else 1,
            key="language_selector",
            on_change=update_language
        )
        
        # Weather Forecast
        st.markdown("#### " + ("天気予報" if st.session_state.language == "Japanese" else "Weather Forecast"))
        
        # Get weather data with caching
        with st.spinner("Loading weather data..." if st.session_state.language == "English" else "天気データを読み込み中..."):
            forecast_data = get_weather_forecast_with_gdd()
        
        # Create and display weather chart
        fig = create_weather_chart(forecast_data, st.session_state.language)
        selected_point = st.plotly_chart(fig, use_container_width=True, key="weather_chart", 
                                        on_select="rerun", selection_mode="points")
        
        # Update selected day based on click
        if selected_point and selected_point.selection.points:
            st.session_state.selected_day = selected_point.selection.points[0]['customdata']
        
        # Display weather data for selected day
        selected_data = forecast_data[st.session_state.selected_day]
        
        # Weather data display - compact horizontal layout
        st.markdown(f"""
            <div class="weather-container">
                <div class="weather-item">
                    <span class="weather-icon">📅</span>
                    <div class="weather-info">
                        <span class="weather-label">{"日付" if st.session_state.language == "Japanese" else "Date"}</span>
                        <span class="weather-value">{selected_data['day']}</span>
                    </div>
                </div>
                <div class="weather-item">
                    <span class="weather-icon">🌡️</span>
                    <div class="weather-info">
                        <span class="weather-label">{"気温" if st.session_state.language == "Japanese" else "Temperature"}</span>
                        <span class="weather-value">{selected_data['temp']}°C</span>
                    </div>
                </div>
                <div class="weather-item">
                    <span class="weather-icon">💧</span>
                    <div class="weather-info">
                        <span class="weather-label">{"湿度" if st.session_state.language == "Japanese" else "Humidity"}</span>
                        <span class="weather-value">{selected_data['humidity']}%</span>
                    </div>
                </div>
                <div class="weather-item">
                    <span class="weather-icon">💨</span>
                    <div class="weather-info">
                        <span class="weather-label">{"風速" if st.session_state.language == "Japanese" else "Wind Speed"}</span>
                        <span class="weather-value">{selected_data['windspeed']} m/s</span>
                    </div>
                </div>
                <div class="weather-item">
                    <span class="weather-icon">📊</span>
                    <div class="weather-info">
                        <span class="weather-label">{"GDD" if st.session_state.language == "Japanese" else "GDD"}</span>
                        <span class="weather-value">{selected_data['accumulated_gdd']}</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Only show pest-related content if processed
        if st.session_state.processed and st.session_state.detected_pests:
            # Show detected pests
            pests_text = ', '.join([p['pest'] for p in st.session_state.detected_pests])
            
            if st.session_state.language == "Japanese":
                label = "検出された害虫："
            else:
                label = "Detected Pests:"
                
            st.markdown(
                f"""
                <div style='background-color: #e3f2fd; 
                           color: #1976d2; 
                           padding: 1rem; 
                           border-radius: 8px; 
                           border: 1px solid #bbdefb;
                           margin-bottom: 1rem;
                           margin-top: 1rem;'>
                    <strong>{label}</strong> {pests_text}
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Quick action buttons - ONLY TWO BUTTONS NOW
            if st.session_state.language == "Japanese":
                st.markdown("**どのようにお手伝いできますか？**")
                btn1_text = "この害虫について教えて"
                btn2_text = "農薬の推奨事項"
            else:
                st.markdown("**How can we assist you?**")
                btn1_text = "Tell me about this pest"
                btn2_text = "Pesticide recommendations"
            
            # Two columns for two buttons
            col_btn1, col_btn2 = st.columns(2)
            
            # Get weather data for chatbot
            try:
                current_weather = getweatherdata(forecast_data[st.session_state.selected_day]['day'])
                if current_weather is None:
                    # Use data from forecast if API fails
                    current_weather = (
                        selected_data['date'],
                        selected_data['temp'],
                        selected_data['humidity'],
                        selected_data['windspeed'],
                        1013,  # default pressure
                        selected_data['description'],
                        selected_data['temp_max'],
                        selected_data['temp_min']
                    )
            except:
                # Fallback to forecast data
                current_weather = (
                    selected_data['date'],
                    selected_data['temp'],
                    selected_data['humidity'],
                    selected_data['windspeed'],
                    1013,
                    selected_data['description'],
                    selected_data['temp_max'],
                    selected_data['temp_min']
                )
            
            current_gdd = forecast_data[st.session_state.selected_day]['accumulated_gdd']
            
            with col_btn1:
                if st.button(btn1_text, use_container_width=True):
                    pest = st.session_state.detected_pests[0]["pest"]
                    pest_info = st.session_state.detected_pests[0]["info"]
                    
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": btn1_text
                    })
                    
                    response = chatbot(
                        pest_info, 
                        st.session_state.chat_history, 
                        btn1_text,
                        current_weather,
                        current_gdd,
                        st.session_state.language
                    )
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    st.rerun()
            
            with col_btn2:
                if st.button(btn2_text, use_container_width=True):
                    pest = st.session_state.detected_pests[0]["pest"]
                    pest_info = st.session_state.detected_pests[0]["info"]
                    
                    # Modified query to include weather conditions
                    if st.session_state.language == "English":
                        query = f"What pesticides do you recommend for {pest}? Also tell me if current weather conditions are suitable for application."
                    else:
                        query = f"{pest}に対してどの農薬を推奨しますか？また、現在の気象条件が散布に適しているか教えてください。"
                    
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": btn2_text  # Show button text in chat
                    })
                    
                    response = chatbot(
                        pest_info, 
                        st.session_state.chat_history, 
                        query,  # Send detailed query to chatbot
                        current_weather,
                        current_gdd,
                        st.session_state.language
                    )
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    st.rerun()
            
            # Chat messages container
            st.markdown("---")
            
            chat_container = st.container(height=400)
            with chat_container:
                for message in st.session_state.chat_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
            
            # Chat input
            placeholder = "メッセージを入力..." if st.session_state.language == "Japanese" else "Type a message..."
            user_question = st.chat_input(placeholder, key="chat_input")
            
            if user_question:
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_question
                })
                
                with st.spinner("考え中..." if st.session_state.language == "Japanese" else "Thinking..."):
                    pest = st.session_state.detected_pests[0]["pest"]
                    pest_info = st.session_state.detected_pests[0]["info"]
                    
                    response = chatbot(
                        pest_info, 
                        st.session_state.chat_history, 
                        user_question,
                        current_weather,
                        current_gdd,
                        st.session_state.language
                    )
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
                
                st.rerun()

# Add footer spacing
st.markdown("<br><br>", unsafe_allow_html=True)
