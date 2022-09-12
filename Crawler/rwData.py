import json

f = open('Crawler/khoa-hoc.json',encoding = 'utf-8-sig')
data = json.load(f)
f.close()

print(len(data))