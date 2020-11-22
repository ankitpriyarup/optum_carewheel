from time import sleep
import serial
ser = serial.Serial('COM3', 115200) # Establish the connection on a specific port

while True:
    bpm_bin = ser.readline()
    dis_bin = ser.readline()
    print(bpm_bin)
    print(dis_bin)
    print("--------------------")
    bpm = (bpm_bin.decode('ascii'))[5:-2]
    dis = (dis_bin.decode('ascii'))[10:-2]
    print(type(bpm))
    print(type(dis))
    p = "BPM is {} and Distance is {}"
    print(p.format(bpm, dis))
    print("*********************")
    