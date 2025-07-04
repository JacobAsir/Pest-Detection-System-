# 🌾 Weather-Based Pest Detection System

An intelligent agricultural pest detection system that combines computer vision, weather data analysis, and AI-powered chatbot assistance to help farmers identify and manage crop pests effectively.

## 🚀 Features

### 🔍 **AI-Powered Pest Detection**
- **YOLOv8-based Detection**: Custom-trained model for accurate pest identification
- **Real-time Image Analysis**: Upload images and get instant pest detection results
- **Multi-pest Recognition**: Detects various agricultural pests with confidence scores
- **Visual Annotations**: Clear bounding boxes and labels on detected pests

### 🌤️ **Weather Integration**
- **Real-time Weather Data**: Current weather conditions and 7-day forecasts
- **Growing Degree Days (GDD)**: Automatic calculation for pest development tracking
- **Spray Condition Analysis**: Evaluates optimal conditions for pesticide application
- **Interactive Weather Charts**: Visual representation of temperature, humidity, and wind data

### 🤖 **Intelligent Chatbot Assistant**
- **Pest Information**: Detailed information about detected pests
- **Treatment Recommendations**: AI-powered pesticide and treatment suggestions
- **Weather-aware Advice**: Recommendations based on current weather conditions
- **Multilingual Support**: Available in English and Japanese
- **Context-aware Responses**: Considers pest type, weather, and GDD data

### 🌍 **User-Friendly Interface**
- **Streamlit Web App**: Clean, responsive web interface
- **Drag & Drop Upload**: Easy image upload functionality
- **Real-time Processing**: Instant results with progress indicators
- **Mobile Responsive**: Works on desktop and mobile devices

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Machine Learning**: YOLOv8 (Ultralytics), PyTorch
- **AI Chatbot**: LangChain + Groq (Llama 3.1)
- **Weather API**: OpenWeatherMap
- **Image Processing**: OpenCV, Pillow
- **Data Visualization**: Plotly
- **Geolocation**: Geopy

## 📋 Prerequisites

- Python 3.8 or higher
- GROQ API key (for chatbot functionality)
- OpenWeatherMap API key (for weather data)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/pest-detection.git
   cd pest-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   WEATHER_API_KEY=your_openweathermap_api_key_here
   ```

4. **Download the trained model**
   - Ensure `best.pt` (YOLOv8 model) is in the root directory
   - The model is trained specifically for agricultural pest detection

## 🎯 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface**
   - Open your browser and go to `http://localhost:8501`

3. **Upload an image**
   - Drag and drop or browse to select an image of crops/plants
   - Supported formats: JPG, JPEG, PNG

4. **Get results**
   - View detected pests with bounding boxes
   - Check weather conditions and spray recommendations
   - Ask the AI assistant for detailed advice

## 📁 Project Structure

```
pest-detection/
├── app.py                 # Main Streamlit application
├── infer.py              # Pest detection inference logic
├── chat.py               # Chatbot and weather integration
├── best.pt               # Trained YOLOv8 model
├── requirements.txt      # Python dependencies
├── Pest_Detection.ipynb  # Model training notebook
├── .env                  # Environment variables (create this)
└── README.md            # Project documentation
```

## 🔧 Configuration

### Weather Settings
- Default location: Tokyo (if geolocation fails)
- Temperature units: Celsius
- Wind speed units: m/s
- Forecast period: 7 days

### Model Settings
- Confidence threshold: 0.25
- IoU threshold: 0.45
- Maximum detections: 10

### Spray Condition Thresholds
- Optimal temperature: 15-32°C
- Maximum wind speed: 4.5 m/s
- Humidity considerations: 40-85%

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics** for the YOLOv8 framework
- **Roboflow** for dataset management and training infrastructure
- **OpenWeatherMap** for weather data API
- **Groq** for fast AI inference capabilities

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check the documentation in the code comments
- Review the Jupyter notebook for model training details

## 🔮 Future Enhancements

- [ ] Mobile app development
- [ ] Additional pest species support
- [ ] Historical pest tracking
- [ ] Integration with farm management systems
- [ ] Offline mode capabilities
- [ ] Advanced analytics dashboard

---

**Made with ❤️ for sustainable agriculture**
