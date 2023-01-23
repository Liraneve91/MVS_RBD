import os
import sys
from SharedCode.PY.utilsScript import get_mac_address
import cv2

print("Python  Version {}".format(sys.version))
print('mac {}'.format(get_mac_address()))
print('Working dir: {}'.format(os.getcwd()))
print('open cv version {}'.format(cv2.getVersionString()))
MAX_PORTS = 4
print('checking for cameras on ports 0 to {}:'.format(MAX_PORTS - 1))
for i in range(MAX_PORTS):
    camera_feed = cv2.VideoCapture(i)
    if camera_feed.isOpened():
        print('\tCamera detected on port {}'.format(i))
        camera_feed.release()
    else:
        print('\tNO cam on port {}'.format(i))
