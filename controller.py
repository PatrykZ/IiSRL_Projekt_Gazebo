'''

Indentyfikacja i sterowanie robotami latającymi
Projekt: Budowa złożonego świata w symulatorze Gazebo i wykonanie w nim zleconej misji 
Adam Gniady
Patryk Zapłata

'''

from tello_msgs.srv import TelloAction
from tello_msgs.msg import TelloResponse
from tello_interface.srv import TelloState
from std_msgs.msg import Empty
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8

import rclpy
from rclpy.node import Node

from .pid import UavPID
from scipy.spatial.transform import Rotation
import time
from enum import Enum
import numpy as np
import math

import darknet as dn
import cv2
from cv_bridge import CvBridge, CvBridgeError

#### Klasa Controller 

class ControllerNode(Node):
    class TelloState(Enum):
        LANDED = 1
        TAKINGOFF = 2
        HOVERING = 3
        FLYING = 4
        LANDING = 5
        NONE = 0

    state = TelloState.LANDED
    next_state = TelloState.NONE
    action_done = False
    g2rr = True

    ## Parametry regulatora PID
    kpxy = 0.07
    kixy = 0.000003
    kdxy = 0.83
    reg = UavPID(kp=[kpxy, kpxy, 0.04, 1.0], ki=[kixy, kixy, 0.000004, 1.0], kd=[kdxy, kdxy, 0.9, 1.0])
    
    ## Zmienna określająca szukany przez dron obiekt
    search = "person"
    search_bool = False
    


    
    def __init__(self):
        super().__init__('controller_node')
        
        self.tello_controller = self.create_subscription(Empty, '/iisrl/tello_controller', self.main_callback, 10)
        self.tello_response = self.create_subscription(TelloResponse, '/drone1/tello_response',
                                                       self.tello_response_callback, 10)
        self.tello_service_server = self.create_service(TelloState, '/iisrl/tello_state', self.state_callback)
        self.tello_service_client = self.create_client(TelloAction, '/drone1/tello_action')

        ## Odczytywanie i wysyłanie danych z węzła 2grr 
        self.tello_g2rr = self.create_subscription(Odometry, '/republisher/tello_1/odom', self.tello_g2rr, 10)
        self.tello_g2rr_pub_x = self.create_publisher(Float32, 'iisrl/pose/x',10)
        self.tello_g2rr_pub_y = self.create_publisher(Float32, 'iisrl/pose/y', 10)
        self.tello_g2rr_pub_z = self.create_publisher(Float32, 'iisrl/pose/z', 10)
        self.tello_g2rr_pub_yaw = self.create_publisher(Float32, 'iisrl/pose/yaw', 10)
        
        ## Pobieranie obrazów z kamery drona
        self.tello_camera = self.create_subscription(Image, 'drone1/image_raw', self.tello_yolo, 10)

        self.service_request = TelloAction.Request()

        ## Pobieranie sieci Darknet oraz bazy obiektów
        dn.set_gpu(0)
        self.net = dn.load_net(b"/home/tello/darknet/cfg/yolov3.cfg", b"/home/tello/darknet/yolov3.weights", 0)
        self.meta = dn.load_meta(b"/home/tello/darknet/cfg/coco.data")
        self.bridge=CvBridge()
        
        ## Zmienne pomocnicze określające 
        ## Obrót - zadany kąt obrotu (1 - 90, 2 - 180, 3 - 270, 4 - 360)
        ## Kierunek (1 - lot w przód, 2 - obracanie drona)
        self.kierunek = 1
        self.obrot = 1
        
###############################################################################
###############################################################################
   
    
    def state_callback(self, request, response):
        response.state = str(self.state)
        response.value = int(self.state.value)

        return response
        
    ## Uruchomienie węzła
    def main_callback(self, msg):
        self.get_logger().info("Uruchomiono wezeł")
        self.action_done = False  # False, gdy istnieje misja do wykonania; True, gdy testujemy start i ladowanie

        self.state = self.TelloState.LANDED
        self.next_state = self.TelloState.TAKINGOFF

        self.controller()

    def controller(self):
        if self.state == self.TelloState.LANDED and self.next_state == self.TelloState.TAKINGOFF:
            self.taking_off_func()

        if self.state == self.TelloState.HOVERING:
            if self.action_done:
                self.action_done = False
                self.landing_func()
            else:
                self.flying_func()
    
    ## Odpowiedz na temat stanu drona
    def tello_response_callback(self, msg):
        if msg.rc == 1:
            self.state = self.next_state
            self.next_state = self.TelloState.NONE

        self.controller()

    ## Start drona
    def taking_off_func(self):
        self.state = self.TelloState.TAKINGOFF
        self.next_state = self.TelloState.HOVERING

        # start drona
        while not self.tello_service_client.wait_for_service(
                timeout_sec=1.0):                    self.get_logger().info("Oczekuje na dostepnosc uslugi Tello...")

        self.service_request.cmd = 'takeoff'
        self.tello_service_client.call_async(self.service_request)
        
    ## Testowanie drona    
    def mission_test(self):
        self.action_done = True
        self.controller()


    ## Lądowanie dronem 
    def landing_func(self):
        self.state = self.TelloState.LANDING
        self.next_state = self.TelloState.LANDED

        # opis procedury ladowania
        ###
        while not self.tello_service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Oczekuje na dostepnosc uslugi Tello...")

        self.service_request.cmd = 'land'
        self.tello_service_client.call_async(self.service_request)    

###############################################################################
###############################################################################

    
    ## Przetważanie tablicy pobieranej z OpenCV na obraz
    def array_to_image(self, arr):
        
        arr = arr.transpose(2, 0, 1)
        c, h, w = arr.shape[0:3]
        arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
        data = arr.ctypes.data_as(dn.POINTER(dn.c_float))
        im = dn.IMAGE(w, h, c, data)
        return im, arr

    ## Funkcja wykrywania obiektów
    def detect(self, net, meta, image, thresh=.5, hier_thresh=.5, nms=0.45):  

        im, image = self.array_to_image(image)
        dn.rgbgr_image(im)
        num = dn.c_int(0)
        pnum = dn.pointer(num)
        dn.predict_image(net, im)
        dets = dn.get_network_boxes(net, im.w, im.h, thresh,
                                    hier_thresh, None, 0, pnum)
        num = pnum[0]
        if nms: dn.do_nms_obj(dets, num, meta.classes, nms)

        res = []
        for j in range(num):
            a = dets[j].prob[0:meta.classes]
            if any(a):
                ai = np.array(a).nonzero()[0]
                for i in ai:
                    b = dets[j].bbox
                    res.append((meta.names[i], dets[j].prob[i],
                                (b.x, b.y, b.w, b.h)))

        res = sorted(res, key=lambda x: -x[1])
        if isinstance(image, bytes): dn.free_image(im)
        dn.free_detections(dets, num)
        return res

    ## Przetwarzanie obrazu pobranego z kamery drona
    def tello_yolo(self, msg):

        try:
            img=self.bridge.imgmsg_to_cv2(msg,"bgr8")
            width = int(96)
            height = int(96)
            dim=(width, height);
            self.resized = cv2.resize(img, dim)
            self.cv_image=self.resized
        except CvBridgeError as e:
            print(e)


    ## Funkcja pobierająca pozycje i kąty obrotu drona
    def tello_g2rr(self,msg):
     

        ##Pobieranie danych
        
        self.pos_x=msg.pose.pose.position.x
        self.pos_y=msg.pose.pose.position.y
        self.pos_z=msg.pose.pose.position.z

        self.Qx = msg.pose.pose.orientation.x
        self.Qy = msg.pose.pose.orientation.y
        self.Qz = msg.pose.pose.orientation.z
        self.Qw = msg.pose.pose.orientation.w

        self.rot=Rotation.from_quat([self.Qx,self.Qy,self.Qz,self.Qw])
        self.roll, self.pitch, self.yaw = self.rot.as_euler('xyz')
        self.pos_x_loc = self.pos_x * math.cos(self.yaw)+self.pos_y*math.sin(self.yaw)
        self.pos_y_loc = -self.pos_x * math.sin(self.yaw) + self.pos_y * math.cos(self.yaw)

        ## Wysyłanie danych
        
        msg_x=Float32()
        msg_x.data = msg.pose.pose.position.x

        msg_y=Float32()
        msg_y.data = msg.pose.pose.position.y

        msg_z=Float32()
        msg_z.data=msg.pose.pose.position.z

        msg_yaw=Float32()
        msg_yaw.data = self.yaw
        
        
        self.tello_g2rr_pub_x.publish(msg_x)
        self.tello_g2rr_pub_y.publish(msg_y)
        self.tello_g2rr_pub_z.publish(msg_z)
        self.tello_g2rr_pub_yaw.publish(msg_yaw)

        ## Lot w góre drona na zadaną wysokość 

        if self.kierunek == 1:
            self.reg.insert_error_value_pair('Z', 1, self.pos_z)


    ## Rozpoczęcie wykonywanej misji
    def flying_func(self):
        self.state = self.TelloState.FLYING
        self.next_state = self.TelloState.FLYING

        if self.action_done:
            self.next_state = self.TelloState.HOVERING
        else:

            self.service_request.cmd = 'rc 0 0 ' + str(self.reg.calculate('Z')) + ' 0'
            self.get_logger().info("Start flying up")
            

            self.start_yaw = 0

            self.tello_service_client.call_async(self.service_request)
            self.timer = self.create_timer((0.033), self.mission_func)

    ## Misja do wykonania
    def mission_func(self):


        while not self.tello_service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Oczekuje na dostepnosc uslugi Tello...")


        if self.pos_z >= 1:

            ## Lot prosto i detekcja
            
            if self.kierunek == 1: 

                self.service_request.cmd = 'rc 0.033 0 0 0'
                self.tello_service_client.call_async(self.service_request)
                
                frame = np.array(self.cv_image, dtype=np.uint8)
                r = self.detect(self.net, self.meta, frame)
                if r:
                    print("Wykryto: " + str(len(r)) +" obiekt/y")
                    #print(r)
                    temp = str(r).split(',')
                    for i in range(len(r)):
                        r_s = np.empty(len(r),dtype=(np.str,16))
                        r_nazwa = np.empty(len(r),dtype=(np.str,16))
                        r_procent = np.empty(len(r))
                        r_x = np.empty(len(r))
                        r_y = np.empty(len(r))
                        r_w = np.empty(len(r))
                        r_h = np.empty(len(r))
                        r_odleglosc = np.empty(len(r))

                        temp2 = str(temp[6*i]).strip("[(' ")
                        r_nazwa[i] = temp2[1:].strip("'")
                        r_procent[i] = float(temp[6*i+1].strip('('))
                        r_x[i] = float(temp[6*i+2].strip()[1:])
                        r_y[i] = float(temp[6*i+3].strip())
                        r_w[i] = float(temp[6*i+4].strip())
                        r_h[i] = float(temp[6*i+5].strip(")] "))
                        print("Nr obiektu: "+ str(i))
                        print("Obiekt: " +str(r_nazwa[i]))
                        print("Prawdopodobieństwo: " +str(r_procent[i]))
                        
                        ## Przetarzanie obrazu na odleglosc
                        r_odleglosc[i] = (((2 * 3.14 * 180) / (r_w[i] + r_h[i] * 360) * 1000 + 3)/4) - 13
                        print("Odległość od obiektu: " +str(r_odleglosc[i]))
                        print("")

                    ## self.search_bool = True - wykryto szukany obiekt
                    ## self.search_bool = False - dalsze szukanie
                    for j in range(len(r_nazwa)):
                        if r_nazwa[j]==self.search:
                            self.search_bool = True
                            self.odleglosc = r_odleglosc[j]
                        else:
                            self.kierunek = 2
                            self.search_bool = False


            ## Ladowanie przy obiekcie
            if self.search_bool and self.odleglosc <= 3:

                self.service_request.cmd = 'rc 0 0 0 0'
                self.get_logger().info("Mission passed")
                self.action_done = True
                self.timer.destroy()
                self.state = self.TelloState.HOVERING
                self.controller()
                
            elif self.search_bool and self.odleglosc >4:
                self.search_bool = False  
                    
            
            ## Obrot drona co 90 stopni
            elif self.kierunek == 2 and self.obrot == 1: ##### OBROT #####

                self.service_request.cmd = 'rc 0 0 0 0.1'
                self.tello_service_client.call_async(self.service_request)    
                if self.yaw >= 2*math.pi/4:

                   self.kierunek = 1 
                   self.obrot = 2

            elif self.kierunek == 2 and self.obrot == 2: ##### OBROT #####

                self.service_request.cmd = 'rc 0 0 0 0.1'
                self.tello_service_client.call_async(self.service_request)    

                if self.yaw >= 3.10:
                    self.kierunek = 1                   
                    self.obrot = 3
                    
            elif self.kierunek == 2 and self.obrot == 3: ##### OBROT #####

                self.service_request.cmd = 'rc 0 0 0 0.1'
                self.tello_service_client.call_async(self.service_request)    

                if self.yaw >= -2*math.pi/4:
                    self.kierunek = 1                   
                    self.obrot = 4 
                    
            elif self.kierunek == 2 and self.obrot == 4: ##### OBROT #####

                self.service_request.cmd = 'rc 0 0 0 0.1'
                self.tello_service_client.call_async(self.service_request)    

                if self.yaw >= -0.10:
                    self.kierunek = 1                   
                    self.obrot = 1 

      

def main(args=None):
    rclpy.init()

    cn = ControllerNode()

    rclpy.spin(cn)
    cn.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
