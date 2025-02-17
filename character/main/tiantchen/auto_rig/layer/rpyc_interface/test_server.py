import json, requests    

json_data = {"folder": "/aigc_cfs_2/xiaqiangdai/project/objaverse_retrieve/data/generate/4e7d59a4-cde2-547f-bbeb-1168bdf1dd18"}
json_data = json.dumps(json_data)
headers = {"Content-Type": "application/json"}
res = requests.post(
    "http://url:8080/app_autoRig_layer/combine",
    data=json_data,
    headers=headers,
)

print(res)