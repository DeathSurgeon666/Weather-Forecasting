import requests # This libraray helps us to fetch data from API

import pandas as pd #for handling and analysing data

import numpy as np #for numerical operations

from sklearn.model_selection import train_test_split #to split data into training and testing sets

from sklearn.preprocessing import LabelEncoder #to convert catogerical data into numericals values

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #models for classification and regression t

from sklearn.metrics import mean_squared_error #to measure the accuracy of our predictions

from datetime import datetime, timedelta #to handle date and time
import pytz

API_KEY = 'e856d2bef199980c4aa5f47e1b09acd1' #replace with your actual API key

BASE_URL = 'https://api.openweathermap.org/data/2.5/' #base url for making API requests
# Define your function here, for example:
def get_weather_data(city):
	url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
	response = requests.get(url) #send the get request to API 
	data = response.json() #convert the response to json
	return {
		'city': data['name'],
		'current_temperature': round(data['main']['temp']),
		'feels_like': round(data['main']['feels_like']),  
		'humidity': round(data['main']['humidity']), 
		'temp_min': round(data['main']['temp_min']),
		'temp_max': round(data['main']['temp_max']),
		'description':data['weather'][0]['description'],
		'country': data['sys']['country'],
		'wind_gust_dir' : data['wind']['deg'],
		'pressure': round(data['main']['pressure']),
		'wind_gust_speed': round(data['wind']['speed']),
	}
def read_historical_data(csv_file):
	# Read the historical data from a CSV file
	df = pd.read_csv(csv_file)
	df = df.dropna()  # Drop rows with missing values
	df = df.drop_duplicates()  # Drop duplicate rows
	return df  

def prepare_data(data):
	le = LabelEncoder()  # Initialize the label encoder
	data['WindGustDir'] = le.fit_transform(data['WindGustDir'])  # Encode categorical data
	data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])  # Encode categorical data
	x = data[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]
	y = data['RainTomorrow']  # Target variable
	return x, y, le

def train_rain_model(x, y):
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

	model = RandomForestClassifier(n_estimators=100, random_state=42)

	model.fit(x_train, y_train) #train the model

	y_pred = model.predict(x_test) #to make predictions on test set

	print("Mean Squared Error for Rain Model")

	print(mean_squared_error(y_test, y_pred))

	return model

def prepare_regression_data(data, feature):
	x, y = [ ] ,[] 
	for i in range(len(data) - 1):
		x.append(data[feature].iloc[i]) #append the feature value to x
		y.append(data[feature].iloc[i + 1]) #append the next value of the feature to y 

	x = np.array(x).reshape(-1, 1) #reshape x to 2D array	
	y = np.array(y) #convert y to numpy array
	return x , y 

def predict_future(model, current_value):
	predictions = [current_value]

	for i in range(5):
		next_value = model.predict(np.array([[predictions[-1]]]))
		predictions.append(next_value[0])

	return predictions[1:]

def train_regression_model(x, y):
	model = RandomForestRegressor(n_estimators=100, random_state=42)
	model.fit(x, y)
	return model

def weather_view():
	city = input("Enter the city name: ")
	current_weather = get_weather_data(city)
	#load historical data
	historical_data = read_historical_data('weather.csv')
	#prepare and train the rain prediction model 
	x, y, le = prepare_data(historical_data)

	rain_model = train_rain_model(x, y)

	wind_deg = current_weather.get('wind_gust_dir', 0)  # Use value from current_weather or default to 0

	compass_points = [
		("Ν", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25), ("ΕΝΕ", 56.25, 78.75), ("Ε", 78.75, 101.25),
		("ESE", 101.25, 123.75), ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
		("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75), ("W", 258.75, 281.25),
		("WNW", 281.25, 303.75), ("MW", 303.75, 326.25), ("NNW", 326.25, 348.75)
	]

	compass_direction = next(point for point, start, end in compass_points if start <= wind_deg < end)

	# Decode the compass direction using le, which is defined above
	compass_direction_encoded = le.inverse_transform([compass_direction])[0] if compass_direction in le.classes_ else -1

	current_data = {
		'MinTemp': current_weather['temp_min'],
		'MaxTemp': current_weather['temp_max'],
		'WindGustDir': compass_direction_encoded,  # Placeholder, you can set it to a specific value
		'WindGustSpeed': current_weather.get('wind_gust_speed', 0),  # Use the correct key and provide a default
		'Humidity': current_weather['humidity'],
		'Pressure': current_weather['pressure'],  # Placeholder, you can set it to a specific value
		'Temp': current_weather['current_temperature'],
	}

	current_df = pd.DataFrame([current_data])
	#rain prediction 
	rain_prediction = rain_model.predict(current_df)[0]
	#prepare regression model for temparature and humidity 
	x_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
	x_hum , y_hum = prepare_regression_data(historical_data, 'Humidity')
	temp_model = train_regression_model(x_temp, y_temp)	
	hum_model = train_regression_model(x_hum, y_hum)
	#predict future temperature and humidity 
	future_temp = predict_future(temp_model, current_weather['temp_min'])
	future_hum = predict_future(hum_model, current_weather['humidity'])				
	#prepare time for future predictions
	timezone = pytz.timezone('Asia/Kolkata')  # Replace with your desired timezone
	now = datetime.now(timezone)
	next_hour = now + timedelta(hours=1)
	next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
	future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]
	#display the results
	print(f"City: {city}, {current_weather['country']}")
	print(f"Current Temperature: {current_weather['current_temperature']}°C")
	print(f"Feels Like: {current_weather['feels_like']}°C")
	print(f"Minimum Temperature: {current_weather['temp_min']}°C")
	print(f"Maximum Temperature: {current_weather['temp_max']}°C")
	print(f"Humidity: {current_weather['humidity']}%")
	print(f"Weather Prediction: {current_weather['description']}")
	print(f"Rain Prediction: {'Yes' if rain_prediction else 'No'}")

	print("\nFuture Temperature Predictions:")

	for time, temp in zip(future_times, future_temp):
		print(f" {time}: {round(temp, 1)}°C")

	print("\nFuture Humidity Predictions:")
	for time, hum in zip(future_times, future_hum):
		print(f" {time}: {round(hum, 1)}%")
weather_view()

