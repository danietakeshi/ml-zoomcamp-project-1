import requests
import sys

if len(sys.argv) == 1:
    url = 'http://0.0.0.0:9696/predict'
else:
    url = f'{sys.argv[1]}/predict'

crab_info_1 = {
    'height': 0.14478,
     'weight': 1.199111356806713,
     'density': 39.376091640190154,
     'bmi': 57.20610404133722,
     'diameter': 0.40005,
     'length': 0.52578,
     'sex': 'F'
}

crab_info_2 = {
    'height': 0.045720000000000004,
     'weight': 0.07956569994897092,
     'density': 58.42414401372047,
     'bmi': 38.06391406637903,
     'diameter': 0.14478,
     'length': 0.20574000000000003,
     'sex': 'I'
}

r_1 = requests.post(url, json=crab_info_1).json()
r_2 = requests.post(url, json=crab_info_2).json()

print(f"Response 1: {r_1}")
print(f"Response 2: {r_2}")