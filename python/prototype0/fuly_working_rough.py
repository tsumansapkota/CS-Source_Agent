import win32gui
import win32ui
import win32con
import time
import numpy as np
import cv2
import PIL.Image as Image
import socket
import struct
import os
from threading import Thread

# for prientscin
windowname = 'Counter-Strike Source'
lshift = 3
tshift = 26
fps = 0
last_time = 0

# for predictionTHread
fileNumber = 0
np_image_data = None
predictIt = False
isEnemy = False
thistime = 0


class DataReceiver():

    def __init__(self):
        programdir = r"F:\Programmers\[C and C++]\VStudio\CShack\CShack\Release"
        os.chdir(programdir)
        os.system("start CShack.exe")
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        # for socket programming
        self.s = socket.socket()
        host = '127.0.0.1'
        port = 2000
        self.s.connect((host, port))
        time.sleep(1)
        print("Connected to IP - " + host)

        self.pause_thread = True
        self.stop_thread = False

        self.allPoints = []
        self.drawPoints = []  # draw points taken from all points
        self.allPlayers = []
        self.drawPlayers = []  # taken from all Players
        self.renewValue = False
        self.myPlayer = []
        return

    def renew_info(self):
        self.renewValue = True

    def receiver(self, _):
        dps = 0
        last_received = time.time()

        print('receiving thread started')

        while not self.stop_thread:
            while self.pause_thread:
                time.sleep(0.01)
                pass
            # ______________________
            if self.renewValue:
                self.renewValue = False
                self.drawPoints.clear()
                for points in self.allPoints:
                    self.drawPoints.append(points)
                self.drawPlayers.clear()
                for players in self.allPlayers:
                    self.drawPlayers.append(players)
            self.allPoints.clear()
            self.allPlayers.clear()
            try:
                self.myPlayer = self.drawPlayers[0][:6]
            except IndexError:
                print(len(self.drawPlayers))
                print('index error in allPlayers')
            self.allPlayers.clear()

            dps += 1

            while len(self.s.recv(1024)) != 1:  # begining of data sending
                pass

            try:
                data = self.s.recv(1024)  # consists of: number of box info && number of team info
                loopData = struct.unpack('ii', data[0:8])
                # print('Loopdata, ', loopData)

                data = self.s.recv(1024)
                for num in range(loopData[0]):
                    # self.allPoints.append(struct.unpack('iiii', data[0:16]))  # four corners of the box
                    self.allPoints.append(
                        struct.unpack('iiiii', data[0:20]))  # four corners of the box and player index || updated
                    data = data[20:]  # delete previous box

                for num in range(loopData[1]):
                    #   DWORD CBaseEntity;
                    # 	int Team;
                    # 	int Health;
                    # 	float Position[3];
                    # 	float AimbotAngle[3];
                    self.allPlayers.append(struct.unpack('Liiffffff', data[0:36]))
                    # print(self.allPlayers[len(self.allPlayers) - 1])
                    data = data[36:]  # delete previous info



            except struct.error:
                print(struct.error, "error occoured!!")
                continue
            finally:
                pass
            if time.time() - last_received > 1.0:
                print('\nDataPS {}'.format(dps))
                last_received = time.time()
                dps = 0
        # _______________________
        self.s.close()
        pass

    def run_listener(self):
        self.thread = Thread(target=self.receiver, args=(0,), daemon=True)
        self.thread.start()
        self.pause_thread = False
        pass

    def pause_unpause(self, pause=True):
        self.pause_thread = pause

    def stop_listener(self):
        self.stop_thread = True
        print('STOP signal sent| waiting for thread to quit!')
        self.thread.join()


dr = DataReceiver()
dr.run_listener()
time.sleep(0.1)

mapImage = cv2.imread('../dust2.jpg')

while True:

    hdesktop = win32gui.FindWindow(None, windowname)
    left, top, right, bot = win32gui.GetClientRect(hdesktop)
    width = right - left
    height = bot - top
    top += tshift
    left += lshift

    desktop_dc = win32gui.GetWindowDC(hdesktop)
    img_dc = win32ui.CreateDCFromHandle(desktop_dc)
    mem_dc = img_dc.CreateCompatibleDC()
    screenshot = win32ui.CreateBitmap()
    screenshot.CreateCompatibleBitmap(img_dc, width, height)
    mem_dc.SelectObject(screenshot)
    mem_dc.BitBlt((0, 0), (width, height), img_dc, (left, top), win32con.SRCCOPY)

    im = Image.frombuffer(
        'RGB',
        (width, height),
        screenshot.GetBitmapBits(True), 'raw', 'RGBX', 0, 1)
    if im.size[0] < 128:
        continue

    image = cv2.cvtColor(np.array(im), cv2.COLOR_RGBA2RGB)
    binary = image.copy()
    binary[0:height, 0:width] = [0, 0, 0]

    mem_dc.DeleteDC()
    win32gui.DeleteObject(screenshot.GetHandle())

    if time.time() - last_time > 1:
        print('FPS {}'.format(fps))
        last_time = time.time()
        fps = 0

    fps += 1
    try:
        # print(len(dr.drawPoints))
        if len(dr.drawPoints) > 0:
            # print(drawPoints)
            drawPoints = dr.drawPoints
            for i in range(0, len(drawPoints)):
                # if i == 0:
                if i >= 0:
                    if drawPoints[i][0] < 0:
                        drawPoints[i] = (0, drawPoints[i][1], drawPoints[i][2] + drawPoints[i][0], drawPoints[i][3])
                    elif drawPoints[i][0] + drawPoints[i][2] > width:
                        drawPoints[i] = (drawPoints[i][0], drawPoints[i][1], width - drawPoints[i][0], drawPoints[i][3])

                    if drawPoints[i][1] < 0:
                        drawPoints[i] = (drawPoints[i][0], 0, drawPoints[i][2], drawPoints[i][3] + drawPoints[i][1])
                    elif drawPoints[i][1] + drawPoints[i][3] > width:
                        drawPoints[i] = (
                            drawPoints[i][0], drawPoints[i][1], drawPoints[i][2], height - drawPoints[i][1])

                    if drawPoints[i][2] / drawPoints[i][3] < 0.1 or drawPoints[i][3] / drawPoints[i][2] < 0.25:
                        drawPoints.remove(drawPoints[i])
                        continue

                    row_start = drawPoints[i][1]
                    row_end = drawPoints[i][1] + drawPoints[i][3]
                    col_start = drawPoints[i][0]
                    col_end = drawPoints[i][0] + drawPoints[i][2]
                    roi = image[row_start:row_end, col_start:col_end]
                    binary[row_start:row_end, col_start:col_end] = roi

                    cv2.putText(binary, "{}".format(drawPoints[i][4]), (col_start, row_start - 1),
                                cv2.FONT_HERSHEY_PLAIN,
                                1, (200, 200, 200))

                    # thistime = time.time()
                    np_image_data = cv2.imencode('.jpg', roi)[1].tostring()
                    predictIt = True

                    # if predThread is None:
                    #     predThread = threading.Thread(target=predFunc, args=(0,))
                    #     predThread.start()
    except Exception as e:
        print('-->', e, '\n__**__')

    # if isEnemy:
    #     cv2.putText(binary, "ENEMY", (300, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200))
    cv2.imshow("finalWindow", binary)

    temp = mapImage.copy()
    for players, index in zip(dr.drawPlayers, range(len(dr.drawPlayers))):
        centrex = int((players[4] + 896) / 7.55 + 21)
        centrey = int((players[3] + 2191.96875) / 7.55 + 10)
        if index == 0:
            cv2.circle(temp, (centrex, centrey), 3, (255, 0, 0), thickness=3)
        elif players[1] == dr.myPlayer[1]:
            cv2.circle(temp, (centrex, centrey), 3, (0, 255, 0), thickness=3)
        else:
            cv2.circle(temp, (centrex, centrey), 3, (0, 0, 255), thickness=3)
        if players[2] <= 1:
            cv2.circle(temp, (centrex, centrey), 3, (5, 200, 200), thickness=2)

    # if len(dr.myPlayer) > 0:
    #     # -2191.96875, -896
    #     centrex = int((dr.myPlayer[4] + 896) / 7.55 + 21)
    #     centrey = int((dr.myPlayer[3] + 2191.96875) / 7.55 + 10)
    #     print(dr.myPlayer)
    #     print((centrey, centrex))
    #     cv2.circle(temp, (centrex, centrey), 3, (255, 0, 0), thickness=3)

    cv2.imshow('Map', temp)

    dr.renew_info()
    if cv2.waitKey(25) & 0xFF == ord('q'):
        dr.stop_listener()
        cv2.destroyAllWindows()
        os.system("taskkill /F /IM CShack.exe /T")
        # predThread.join()
        # sess.close()
        break
