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
width = 640
height = 480

# for predictionTHread
fileNumber = 0
np_image_data = None
predictIt = False
isEnemy = False
thistime = 0

players = []

class Player():
    """
    Stores all the information of Players of CS:Source
    """

    def __init__(self):
        self.playerName = ''
        self.cbaseEntity = 0
        self.team = 0
        self.health = 0
        self.position = (0., 0., 0.)
        self.aimbotAngle = (0., 0., 0.)

    def is_alive(self):
        """
        If player is alive or not
        :return:bool: True if alive
        """
        return (self.health > 1)

    def is_friend_to(self, myPlayer):
        """
        Returns if the player is friend
        :param myPlayer:Player (object) to be compared to
        :return: True if they belong to same team
        """
        return (self.team == myPlayer.team)


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

        self.totalPlayers = 0
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
                for play_ers in self.allPlayers:
                    self.drawPlayers.append(play_ers)
            self.allPoints.clear()
            self.allPlayers.clear()
            try:
                self.myPlayer = self.drawPlayers[0][:6]
            except IndexError:
                print(len(self.drawPlayers))
                print('index error in allPlayers')
            self.allPlayers.clear()

            dps += 1

            # request data

            while len(self.s.recv(1024)) != 1:  # begining of data sending
                pass

            try:
                data = self.s.recv(4096)  # consists of: number of box info && number of team info
                loopData = struct.unpack('ii', data[0:8])
                data = data[8:]
                for num in range(loopData[0]):
                    # x1,y1, width, height, playerIndex ///here width = height/2
                    unpacked = struct.unpack('iiiii', data[0:20])
                    self.allPoints.append(unpacked)  # four corners of the box and player index || updated
                    data = data[20:]  # delete previous box

                # managing total players in the game
                self.totalPlayers = len(players)
                if self.totalPlayers == loopData[1]:
                    pass
                elif self.totalPlayers < loopData[1]:
                    for i in range(loopData[1] - self.totalPlayers):
                        players.append(Player())
                elif self.totalPlayers > loopData[1]:
                    for i in range(self.totalPlayers - loopData[1]):
                        players.remove(players[len(players) - 1])

                # setting player information
                for num in range(loopData[1]):
                    #   DWORD CBaseEntity;
                    # 	int Team;
                    # 	int Health;
                    # 	float Position[3];
                    # 	float AimbotAngle[3];
                    unpacked = struct.unpack('Liiffffff', data[0:36])
                    # self.allPlayers.append(unpacked)
                    players[num].cbaseEntity = unpacked[0]
                    players[num].team = unpacked[1]
                    players[num].health = unpacked[2]
                    players[num].position = unpacked[3:6]
                    players[num].aimbotAngle = unpacked[6:9]

                    # print(self.allPlayers[len(self.allPlayers) - 1])
                    data = data[36:]  # delete previous info
                    # print(len(players), '=players')


            except struct.error:
                print(struct.error, "error occoured!!")  # doesnt really occuor
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
                    elif drawPoints[i][1] + drawPoints[i][3] > height:
                        drawPoints[i] = (
                            drawPoints[i][0], drawPoints[i][1], drawPoints[i][2], height - drawPoints[i][1])

                    # remove heavily cropped player box
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
    for player, index in zip(players, range(len(players))):
        centrex = int((player.position[1] + 896) / 7.55 + 21)
        centrey = int((player.position[0] + 2191.96875) / 7.55 + 10)
        if index == 0:
            cv2.circle(temp, (centrex, centrey), 3, (255, 0, 0), thickness=3)
        elif player.team == players[0].team:
            cv2.circle(temp, (centrex, centrey), 3, (0, 255, 0), thickness=3)
        else:
            cv2.circle(temp, (centrex, centrey), 3, (0, 0, 255), thickness=3)

        if player.health < 2:
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
