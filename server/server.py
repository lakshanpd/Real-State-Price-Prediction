from flask import Flask, jsonify, request, redirect, url_for, render_template
import pickle
import sklearn
import numpy as np

app = Flask('__main__', template_folder='../client')

model = None
scaler = None
encoder = None

def load_artifacts():
    global model, encoder, scaler, pipeline
    with open('../model/model.pkl', 'rb') as file:
        model = pickle.load(file)
        print(model)
    with open('../model/encoder.pkl', 'rb') as file:
        encoder = pickle.load(file)
        print(encoder)
    with open('../model/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
        print(scaler)

# def pipeline(model, scaler, encoder, bedrooms, bathrooms, sqft_living, sqft_lot, floors, view, sqft_above, yr_built, yr_renovated, city):
#     vector_1 = np.array([bedrooms, bathrooms, floors, view])
#     vector_2 = scaler.transform([[sqft_living, sqft_lot, sqft_above, yr_built, yr_renovated]]).flatten()
#     cities = encoder.categories_[0]
#     vector_3 = np.zeros(len(cities))
#     city_index = np.where(cities == city)[0]
#     vector_3[city_index] = 1
#     vector = np.concatenate((vector_1[:2], vector_2[:2], vector_1[2:], vector_2[2:], vector_3))
#     vector = vector.reshape(1, -1)
#     return np.round(model.predict(vector), 2)

@app.route('/')
def home():
    cities = encoder.categories_[0]
    return render_template('client.html', cities=cities)

@app.route('/submit', methods=['POST'])
def submit():
    print(request.form[:])
    try:
        Bedrooms = int(request.form['bedrooms'])
        Bathrooms = int(request.form['bathrooms'])
        sqft_living = float(request.form['sqft_living'])
        sqft_lot = float(request.form['sqft_lot'])
        floors = int(request.form['floors'])
        view = int(request.form['view'])
        sqft_above = float(request.form['sqft_above'])
        yr_built = int(request.form['yr_built'])
        yr_renovated = int(request.form['yr_renovated'])
        city = request.form['city']

        price = pipeline(Bedrooms, Bathrooms, sqft_living, sqft_lot, floors, view, sqft_above, yr_built, yr_renovated, city)
        print(price)
        return price;
    
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    load_artifacts()
    # if model==None or encoder==None or scaler==None or pipeline==None:
    #     print('There is a None value')
    # else:
    #     print('There is no None value')
    # print(sklearn.__version__)
    app.run()




