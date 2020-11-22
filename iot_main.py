import serial, requests, time, re
from threading import Thread, Event
ser = serial.Serial('/dev/ttyACM0', 115200)

bpm = "None"
dist = "None"
first = False
second = False
if __name__ == "__main__":
    while True:
        data = str(ser.readline())
        cur = re.findall("M: \d+\.*\d+", data)
        if len(cur) > 0:
            bpm = str(cur[0])[3:]
        cur = re.findall("e: \d+\.*\d+", data)
        if len(cur) > 0:
            dist = str(cur[0])[3:]
        if "ER!" in data:
            if "NO" not in data:
                if first is False:
                    first = True
                elif second is False:
                    second = True
                else:
                    requests.post('http://127.0.0.1:5000/api/post_msg/bb411a9c5f6631634a342779b28cc612',
                            json={"content":"User just sent an SOS signal!"})
                    print("DANGER!")
            else:
                first = False
                second = False


        cur = re.findall("e: \d+\.*\d+", data)
        if dist.isdigit():
            requests.post('http://127.0.0.1:5000/api/post_update/bb411a9c5f6631634a342779b28cc612',
                            json={"username": "ankitpriyarup", "key": "qazwsxedc", "distance": int(dist)})
        if bpm.isdecimal():
            requests.post('http://127.0.0.1:5000/api/post_update/bb411a9c5f6631634a342779b28cc612',
                            json={"username": "ankitpriyarup", "key": "qazwsxedc", "hr": float(bpm)})
        print("BPM: " + str(bpm) + ", Distance: " + str(dist))