import requests

url = 'http://localhost:9696/predict'

data = {'url': 'https://storage.googleapis.com/kagglesdsdata/datasets/1731575/2830785/Tire%20Textures/testing_data/cracked/Cracked-1.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20211210%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20211210T172225Z&X-Goog-Expires=345599&X-Goog-SignedHeaders=host&X-Goog-Signature=a48f50b3584af3d8079bc7cf36ec2967cb37a5b0496b0bae4538334bc8b842ca0fb03e62c813592dced1ba7a79834c4e2dba63fbe6fe24813da5db2d4c8ac8bb41631ea0351e1268d6cdd7800bd390ae32fab1c686d4a463e7cf38d90933cf736a2e26dcece605268bdc8e6d8a8ab862c46e145f603671a850c02a291c8682a221c4d2e333e815c08fa3c84b93e0d1c976a2f00ca9ca1af928fb000ec2a1f2bd10efacb053fb5ea180504df51c871b7baa2dd01fc9f6daecef64c29a70e4235ba804ce92d2afb0d076f9f67449e47177ff277677a8ac0656153dd91eca1c3854b6200374ef0677b259c3d10762d7e6b3dcc3c0fd51d49a5ff37ecbded6e7dd73'}

result = requests.post(url, json=data).json()
print(result)