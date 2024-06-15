from flask import Flask, render_template, request
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def hello_world():
    return render_template('signin.html')

@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/home')
def home():
    return render_template('homepage.html')

@app.route('/signin')
def signin():
    return render_template('signin.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/predict',methods=['POST'])
def predict():
    passenger_count = int(request.form['passenger_count'])
    time= request.form['pickup_time']
    date= request.form['pickup_date']

    def getLatLong(coordinates):
        parts = coordinates.split(',')
        latitude = float(parts[0])
        longitude = float(parts[1])
        return longitude, latitude

    pickupInput = request.form['pickupInput']
    dropoffInput = request.form['dropoffInput']

    (pickup_longitude,pickup_latitude)=getLatLong(pickupInput)
    (dropoff_longitude,dropoff_latitude)=getLatLong(dropoffInput)

    hours = int(time.split(':')[0])
    minutes = int(time.split(':')[1])

    date_obj = datetime.strptime(date, '%Y-%m-%d')

    # Extract month, year, and day components
    month = int(date_obj.month)
    year = int(date_obj.year)
    date_NUM = int(date_obj.day)
    day = int(date_obj.weekday())

    def haversine_distance(latitude, longitude, latitude2, longitude2):
        radius = 6371 # km
        # 2 is dropoff
        # 1 is pickup
        dlat = np.radians(latitude2 - latitude)
        dlon = np.radians(longitude2 - longitude)
        a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(latitude)) * np.cos(np.radians(latitude2)) * np.sin(dlon/2) * np.sin(dlon/2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance = radius * c

        return distance
    
    distance = haversine_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude)

    features = [ distance, passenger_count,hours, minutes, date_NUM, month, year, day]
    prediction=predict_with_ml(features)

    return render_template('result.html', prediction_text='Predicted fare amount is Rs {}'.format(prediction))

def predict_with_ml(features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    # output = '{0:.{1}f}'.format(prediction[0][1], 2)
    return prediction*83.67

if __name__=='__main__':
    app.run(debug=True)