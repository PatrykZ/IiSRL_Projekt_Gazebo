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




