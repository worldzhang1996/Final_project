import requests

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('/Users/zhangshijie/Desktop/COMP5933/experiment/serving/request.json','r')})
print(dir(resp))
print(resp.text)