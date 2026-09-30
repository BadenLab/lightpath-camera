import pprint
import picamera2

picam2 = picamera2.Picamera2()

pprint.pprint(picam2.sensor_modes)
