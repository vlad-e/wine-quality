import requests

url= 'http://localhost:9696/predict'

wine_values = {     
"fixed acidity":10.3,
"volatile acidity":0.59,
"citric acid":0.42,
"residual sugar": 2.8,
"chlorides":0.09,
"free sulfur dioxide":35.0,
"total sulfur dioxide":73.0,
"density":0.9990000000000001,
"pH":3.28,
"sulphates":0.7,
"alcohol":9.5
}

print("Input values:")
print(wine_values,"\n")
print("Output value:")
print(requests.post(url, json=wine_values).json())