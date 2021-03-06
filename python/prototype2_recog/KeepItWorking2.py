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
import tensorflow as tf

# for prientscin
windowname = 'Counter-Strike Source'
lshift = 3
tshift = 26
width = 640
height = 480

# for predictionTHread
np_image_data = None

players = []
playerBoxes = []


class EnemyPredictor():

    def __init__(self):
        self.X = tf.placeholder(tf.float32, shape=[None, 48, 24, 3])
        self.y = tf.placeholder(tf.float32, shape=[None, 2])
        self.hold_prob = tf.placeholder(tf.float32)

        def init_weights(shape):
            rnd = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(rnd)

        def init_bias(shape):
            rnd = tf.constant(0.1, shape=shape)
            return tf.Variable(rnd)

        def conv2d(x, W):
            # x --> [batch, H, W, nChannels]
            # W --> [filterH, filterW, ChannelsIN, ChannelsOUT]
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
            # x --> [batch, H, W, nChannels]
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        def convolutional_layer(input_x, shape):
            W = init_weights(shape)
            b = init_bias([shape[3]])
            return tf.nn.relu(conv2d(input_x, W) + b)

        def normal_full_layer(input_layer, size):
            input_size = int(input_layer.get_shape()[1])
            W = init_weights([input_size, size])
            b = init_bias([size])
            return tf.matmul(input_layer, W) + b

        conv1 = convolutional_layer(self.X, shape=[4, 4, 3, 24])
        conv1pool = max_pool_2x2(conv1)

        conv2 = convolutional_layer(conv1pool, shape=[4, 4, 24, 48])
        conv2pool = max_pool_2x2(conv2)

        conv2shape = tf.shape(conv2pool)
        conv2flat = tf.reshape(conv2pool, [-1, 12 * 6 * 48])

        fully_layer_one = tf.nn.relu(normal_full_layer(conv2flat, 768))
        fully_one_dropout = tf.nn.dropout(fully_layer_one, keep_prob=self.hold_prob)
        self.output = normal_full_layer(fully_one_dropout, 2)

    def start(self):
        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, "../enemy_or_not/model.ckpt")

    def stop(self):
        self.sess.close()

    def is_enemy(self, image):
        pred = tf.argmax(self.output, 1)
        val = self.sess.run(pred, feed_dict={self.X: [image], self.hold_prob: 1.0})
        return val

    def is_enemy_jpt(self, image):

        return 0


class BoxPlayer():

    def __init__(self, xx=0, yy=0, width=0, height=0, player=-1, distance=0):
        self.xx = xx
        self.yy = yy
        self.width = width
        self.height = height
        self.player = player
        self.distance = distance
        pass


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

        return

    def receiver(self, _):
        dps = 0
        last_received = time.time()

        print('receiving thread started')

        while not self.stop_thread:
            while self.pause_thread:
                time.sleep(0.01)
                pass
            # ______________________

            while len(self.s.recv(1024)) != 1:  # begining of data sending
                pass

            try:
                data = self.s.recv(4096)  # consists of: number of box info && number of team info
                loopData = struct.unpack('ii', data[0:8])
                data = data[8:]

                # managing total boxes in the game
                totalBox = len(playerBoxes)
                totalPlayers = len(players)
                givenBox = loopData[0]
                givenPlayers = loopData[1]

                if totalBox == givenPlayers:
                    pass
                elif totalBox < givenPlayers:
                    for i in range(givenPlayers - totalBox):
                        playerBoxes.append(BoxPlayer())
                elif totalBox > givenPlayers:
                    for i in range(totalBox - givenPlayers):
                        playerBoxes.remove(playerBoxes[len(playerBoxes) - 1])

                boxFilled = []
                for num in range(givenBox):
                    # x1,y1, width, height, playerIndex ///here width = height/2
                    unpacked = struct.unpack('iiiii', data[0:20])

                    # box cropping if out of screen
                    values = [i for i in unpacked]

                    box = BoxPlayer(values[0], values[1], values[3] // 2, values[3], values[4], values[2])
                    playerBoxes[values[4]] = box
                    boxFilled.append(values[4])
                    data = data[20:]  # delete previous box

                # make the box not null if not assigned value
                for i in range(givenPlayers):
                    if i not in boxFilled:
                        playerBoxes[i].player = -1

                # managing total players in the game
                if totalPlayers == givenPlayers:
                    pass
                elif totalPlayers < givenPlayers:
                    for i in range(givenPlayers - totalPlayers):
                        players.append(Player())
                elif totalPlayers > givenPlayers:
                    for i in range(totalPlayers - givenPlayers):
                        players.remove(players[len(players) - 1])

                # setting player information
                for num in range(loopData[1]):
                    unpacked = struct.unpack('Liiffffff', data[0:36])
                    players[num].cbaseEntity = unpacked[0]
                    players[num].team = unpacked[1]
                    players[num].health = unpacked[2]
                    players[num].position = unpacked[3:6]
                    players[num].aimbotAngle = unpacked[6:9]

                    data = data[36:]  # delete previous info


            except struct.error:
                print(struct.error, "error occoured!!")  # doesnt really occuor
                continue

            dps += 1
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


# -------------------------------------------------------------------------------------------------------------

dr = DataReceiver()
dr.run_listener()

enemyPred = EnemyPredictor()
enemyPred.start()

mapImage = cv2.imread('../dust2.jpg')
fps = 0
last_time = 0

hdesktop = win32gui.FindWindow(None, windowname)


def screenshot():
    left, top, right, bot = win32gui.GetClientRect(hdesktop)
    width = right - left
    height = bot - top
    top += tshift
    left += lshift
    try:
        desktop_dc = win32gui.GetWindowDC(hdesktop)
        img_dc = win32ui.CreateDCFromHandle(desktop_dc)
        mem_dc = img_dc.CreateCompatibleDC()
        screenshot = win32ui.CreateBitmap()
        screenshot.CreateCompatibleBitmap(img_dc, width, height)
        mem_dc.SelectObject(screenshot)
        mem_dc.BitBlt((0, 0), (width, height), img_dc, (left, top), win32con.SRCCOPY)
    except win32ui.error as e:
        print(e)
        return None

    im = Image.frombuffer(
        'RGB',
        (width, height),
        screenshot.GetBitmapBits(True), 'raw', 'RGBX', 0, 1)
    if im.size[0] < 128:
        return None

    mem_dc.DeleteDC()
    win32gui.DeleteObject(screenshot.GetHandle())
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGBA2RGB)


try:
    while True:

        image = screenshot()
        if image is None:
            continue

        binary = image.copy()
        binary[0:height, 0:width] = [0, 0, 0]

        if time.time() - last_time > 1:
            print('FPS {}'.format(fps))
            last_time = time.time()
            fps = 0

        fps += 1
        try:
            # print(len(playerBoxes))
            if len(playerBoxes) > 0:
                for i, box in zip(range(len(playerBoxes)), playerBoxes):
                    # if i>1:continue
                    # skip unknown value
                    if box.player == -1:
                        continue
                    # crop player boxes
                    if box.xx < 0:
                        box.width += box.xx
                        box.xx = 0
                    elif box.xx + box.width > width:
                        box.width = width - box.xx
                    if box.yy < 0:
                        box.height += box.yy
                        box.yy = 0
                    elif box.yy + box.height > height:
                        box.height = height - box.yy

                    # skip zero value
                    if box.height < 1 or box.width < 1:
                        continue
                    # skip unproportionate value of the boxes
                    if box.width / box.height < 0.1 or box.height / box.width < 0.25:
                        continue
                    if box.width * box.height > width * height * 0.75:
                        continue

                    row_start = box.yy
                    row_end = box.yy + box.height
                    col_start = box.xx
                    col_end = box.xx + box.width
                    roi = image[row_start:row_end, col_start:col_end]



                    binary[row_start:row_end, col_start:col_end] = roi

                    prediction = enemyPred.is_enemy(cv2.resize(roi, (24, 48), interpolation=cv2.INTER_CUBIC))
                    print(prediction, box.player, box.distance)
                    if prediction[0] == 1:
                        cv2.rectangle(binary, (col_start, row_start), (col_end, row_end), (0, 0, 255), thickness=3)

                    cv2.putText(binary, "{}".format(box.player), (col_start, row_start - 1),
                                cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200))
                    # if cv2.waitKey(25) & 0xFF == ord('v'):
                    # cv2.imwrite("../saved_res/snap{}.jpg".format(int((time.time() % 100000000) * 1000000000)), roi)

        except Exception as e:
            print('-->', e, '\nError While Creating Boxes')

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

        # display image window
        cv2.imshow("finalWindow", binary)
        cv2.imshow('Map', cv2.resize(temp, None, fx=0.5, fy=0.5))

        # time.sleep(0.01)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        pass
except KeyboardInterrupt:
    pass
finally:
    dr.stop_listener()
    cv2.destroyAllWindows()
    enemyPred.stop()
    os.system("taskkill /F /IM CShack.exe /T")
    pass
