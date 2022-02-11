IiSRL_Projekt_Gazebo

Projekt na przedmiot : Identyfikacja i sterowanie robotami latajacymi
Temat: Budowa złożonego świata w symulatorze Gazebo i wykonanie w nim zleconej misji

Autorzy:
- Adam Gniady
- Patryk Zapłata

Biblioteki i Moduły:
- g2rr
- Tello Ros2
- Gazebo
- OpenCv
- Darknet
- Yolov3

Instrukcja uruchomienia:
Stworzenie przestrzeni roboczej:

mkdir -p ~/ws_136642/src
cd ~/ws_136642
colcon build --symlink-install

Następnie przygotować pakiet:

cd ~/ws_136642/src

ros2 pkg create --build-type ament_cmake tello_controller

Pliki darknet.py, controller.py i pid.py należy skopiować do folderu:

~/ws_136642/build/tello_controller/tello_controller

Pobrać plik yolov3.weights ze strony:

https://pjreddie.com/media/files/yolov3.weights

W pliku controller.py w linijce 85 i 86 należy podać ścieżkę do plików yolov3.cfg yolov3.weights i coco.names
W pliku yolov3.cfg wymiar height i width powinnien być ten sam co w pliku controller.py w linijce 212 i 213

Uruchamiamy gazebo ze światem simple.world i węzeł g2rr
Otwieramy nowy terminal i wpisujemy komendy:

source tello_ros_ws/install/setup.bash 

cd ~/ws_136642

source install/setup.bash

ros2 run tello_controller controller

Otwieramy nowy terminal i wpisujemy komendę:

ros2 topic pub /iisrl/tello_controller std_msgs/msg/Empty -1






