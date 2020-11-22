import os, json
raw = 'pichle din ki report'
open('commands/_i', 'w').close()
f = open("commands/_i", "a")
f.write(raw)
f.close()
os.system('sh ./commands/ner_time.sh')
json_res = {}
with open('commands/_o') as fp:
   line = fp.readline()
   while line:
        line = fp.readline()
        line = line.replace('HTTP/1.1 200 OK', '')
        if 'data' in line and 'null' not in line:
            json_res.update(json.loads(line))
print(json_res)