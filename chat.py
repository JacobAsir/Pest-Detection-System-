from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import geopy
import requests
from geopy.geocoders import Nominatim
import os
from dotenv import load_dotenv

load_dotenv()

# Load API keys from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

def getcoordinate(city):
    try:
        geolocator = Nominatim(user_agent="PestDetectionApp")
        location = geolocator.geocode(city)
        if location:
            return (location.latitude, location.longitude)
        else:
            return (35.6762, 139.6503)  # Default to Tokyo
    except Exception as e:
        print(f"Error getting coordinates: {e}")
        return (35.6762, 139.6503)

def getweatherdata(day):
    try:
        if not WEATHER_API_KEY:
            raise ValueError("WEATHER_API_KEY not found in environment variables")
        
        city = 'Tokyo, Japan'
        coordinates = getcoordinate(city)
        if not coordinates:
            raise ValueError("Could not get coordinates for the city")
        
        lat, lon = coordinates
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"

        response = requests.get(forecast_url, timeout=10)
        response.raise_for_status()
        res = response.json()
        
        if 'list' not in res:
            raise ValueError("Invalid API response: missing 'list' field")
        
        day_map = {'Today':0, 'Tomorrow':8, '2 days': 16, '3 days': 24, '4 days': 32}
        if day not in day_map:
            raise ValueError(f"Invalid day: {day}")
        
        idx = day_map[day]
        
        if idx >= len(res['list']):
            raise ValueError(f"Weather data not available for {day}")
        
        current_data = res['list'][idx]
        date = current_data['dt_txt'].split(" ")[0]
        temperature = round(current_data['main']['temp'], 1)
        pressure = current_data['main']['pressure']
        humidity = current_data['main']['humidity']
        description = current_data['weather'][0]['description']
        windspeed = round(current_data['wind']['speed'], 1)
        temp_max = round(current_data['main']['temp_max'], 1)
        temp_min = round(current_data['main']['temp_min'], 1)

        return (date, temperature, humidity, windspeed, pressure, description, temp_max, temp_min)
    
    except Exception as e:
        print(f"Error getting weather data: {e}")
        return None

# Initialize ChatGroq
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

llm = ChatGroq(
    model_name="llama-3.1-8b-instant", 
    groq_api_key=GROQ_API_KEY,
    temperature=0.7,
    max_tokens=1000
)

# Keep the original pest information
class_info_dict = {
    "brown-planthopper": "The brown planthopper (Nilaparvata lugens) is a notorious agricultural pest that primarily affects rice crops. Brown planthoppers feed on the sap of rice plants by piercing and sucking through their needle-like mouthparts, leading to weakened plants and reduced yields. Commonly used classes of insecticides for controlling brown planthoppers include neonicotinoids, pyrethroids, and carbamates. For optimal pesticide application, aim for mild temperatures (60-85°F), low to moderate wind speeds (3-10 mph), and moderate humidity. Apply pesticides in the early morning or late afternoon, avoiding extreme weather conditions and ensuring adherence to specific label instructions for each pesticide",
    "green-leafhopper": "The green-leafhopper is a pest affecting crops like rice. Effective control often involves using neonicotinoid or pyrethroid insecticides. Optimal weather conditions for application include temperatures between 70-85°F, low wind speeds, and moderate humidity.",
    "leaf-folder": "The leaf-folder, a pest in rice cultivation, can be managed with systemic insecticides like neonicotinoids or with Bacillus thuringiensis (Bt) formulations. Apply during calm weather with temperatures around 75-85°F and low wind speeds for best results.",
    "rice-bug": "For controlling rice bugs, insecticides such as pyrethroids or neonicotinoids are effective. Optimal weather conditions for application include temperatures around 80°F, low to moderate wind speeds, and moderate humidity.",
    "stem-borer": "Stem-borers in crops like rice can be controlled with insecticides like carbamates or pyrethroids. Apply during temperatures of 75-85°F, low wind speeds, and moderate humidity for optimal efficacy.",
    "whorl-maggot": "To combat whorl-maggots in crops, use insecticides like neonicotinoids or organophosphates. Apply during temperatures of 70-80°F, low wind speeds, and moderate humidity for the best results.",
}

def evaluate_spray_conditions(temperature, humidity, windspeed):
    """Evaluate if current conditions are suitable for pesticide application"""
    suitable = True
    reasons = []
    
    # Temperature check (in Celsius)
    if temperature < 15:
        suitable = False
        reasons.append(f"temperature too low ({temperature}°C)")
    elif temperature > 32:
        suitable = False
        reasons.append(f"temperature too high ({temperature}°C)")
    
    # Wind speed check (in m/s)
    if windspeed > 4.5:
        suitable = False
        reasons.append(f"wind too strong ({windspeed} m/s)")
    
    # Humidity check
    if humidity < 40:
        reasons.append(f"low humidity ({humidity}%) - add adjuvants")
    elif humidity > 85:
        reasons.append(f"high humidity ({humidity}%) - ensure drying time")
    
    return suitable, reasons

def chatbot(info, history, message, weather_data, gdd, language="English"):
    if weather_data is None:
        return "Weather data is currently unavailable. Please try again later."
    
    date, temperature, humidity, windspeed, pressure, description, temp_max, temp_min = weather_data
    
    # Evaluate spray conditions
    suitable, reasons = evaluate_spray_conditions(temperature, humidity, windspeed)
    
    # Create weather info string with ACTUAL data
    weather_info = f"""Temperature: {temperature}°C, Humidity: {humidity}%, Wind: {windspeed} m/s"""
    
    # Determine the type of question
    is_pest_info_query = any(phrase in message.lower() for phrase in ["tell me about", "what is", "この害虫について"])
    is_pesticide_query = any(phrase in message.lower() for phrase in ["pesticide", "recommend", "農薬", "推奨"])
    
    # Different prompts based on query type
    if language == "Japanese":
        if is_pest_info_query:
            prompt_template = """
            農家に害虫について簡潔に説明してください。
            
            害虫情報: {info}
            
            2-3文で説明してください。技術的すぎないように。
            
            質問: {message}
            """
        elif is_pesticide_query:
            spray_status = "可能" if suitable else "不適"
            prompt_template = f"""
            農薬の推奨について答えてください。
            
            害虫情報: {{info}}
            現在の天候: {{weather_info}}
            散布条件: {spray_status} {' - ' + ', '.join(reasons) if reasons else ''}
            
            以下を含めて3-4文で回答：
            1. 推奨農薬（2-3種類）
            2. 現在散布できるか（実際の天候データに基づいて）
            3. 最適なタイミング
            
            質問: {{message}}
            """
        else:
            prompt_template = """
            農家の質問に簡潔に答えてください。
            
            害虫: {info}
            天候: {weather_info}
            GDD: {gdd}
            
            3-4文以内で実用的な回答を。
            
            質問: {message}
            """
    else:
        if is_pest_info_query:
            prompt_template = """
            Briefly explain this pest to a farmer.
            
            Pest information: {info}
            
            In 2-3 sentences, explain what this pest is and what damage it causes. Keep it simple.
            
            Question: {message}
            """
        elif is_pesticide_query:
            spray_status = "YES" if suitable else "NO"
            prompt_template = f"""
            Answer about pesticide recommendations.
            
            Pest info: {{info}}
            Current weather: {{weather_info}}
            Can spray now: {spray_status} {' - ' + ', '.join(reasons) if reasons else ''}
            
            In 3-4 sentences total, cover:
            1. Recommended pesticides (2-3 options)
            2. Whether you can spray NOW based on the ACTUAL weather data provided
            3. Best timing if not now
            
            Use the exact weather values provided. Do not make up different numbers.
            
            Question: {{message}}
            """
        else:
            prompt_template = """
            Answer the farmer's question concisely.
            
            Context:
            - Pest: {info}
            - Weather: {weather_info}
            - GDD: {gdd}
            
            Provide a practical answer in 3-4 sentences.
            
            Question: {message}
            """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=['info', 'weather_info', 'gdd', 'history', 'message']
    )

    output_parser = StrOutputParser()
    chain = PROMPT | llm | output_parser

    try:
        response = chain.invoke({
            "info": info,
            "weather_info": weather_info,
            "gdd": gdd,
            "history": "",  # Simplified - not using history for conciseness
            "message": message
        })
        return response.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I apologize, but I'm having trouble generating a response. Please try again."
