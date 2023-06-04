import face_recognition
import os, sys
import cv2
import numpy as np
import math
import time

def face_confidence(face_distance, face_match_threshold=0.6) -> str:
    """
    La función calcula el porcentaje de coincidencia entre dos caras en función de su distancia y un
    valor de umbral.

    :param face_distance: La distancia entre dos caras en un sistema de reconocimiento facial. Es una
    medida de cuán diferentes son dos caras entre sí.
    :param face_match_threshold: face_match_threshold es un valor de umbral que determina si una cara se
    considera una coincidencia o no. Si la distancia entre dos caras es menor o igual que
    face_match_threshold, entonces se consideran una coincidencia. De lo contrario, no se consideran una
    coincidencia. El valor predeterminado es 0.6.
    :return: una cadena que representa el porcentaje de qué tan cerca una distancia de cara dada
    coincide con un umbral de coincidencia de cara. Si la distancia entre caras es mayor que el umbral
    de coincidencia de caras, la función devuelve un porcentaje de valor lineal. Si la distancia entre
    caras es menor o igual que el umbral de coincidencia de caras, la función devuelve un porcentaje de
    valor no lineal.
    """
    range_value = 1.0 - face_match_threshold
    linear_value = (1.0 - face_distance) / (range_value * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_value * 100, 2)) + '%'
    else:
        value = (linear_value + ((1.0 - linear_value) * math.pow((linear_value - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


class FaceRecon:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self) -> None:
        self.encode_faces()
        self.last_verification_time = time.time()

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)

        #print(self.known_face_names)

    def run_recon(self):
        video_capture = cv2.VideoCapture(0)  # Dependiendo de la cantidad de cámaras del dispositivo
        if not video_capture.isOpened():
            sys.exit("No se detecta la cámara")

        unknown_start_time = None  # Tiempo de inicio cuando se detecta una cara desconocida
        unknown_detected = False  # Indicador de si se ha detectado una cara desconocida

        while True:
            ret, frame = video_capture.read()
            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Se redimensiona para mejorar rendimiento de CPU
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # Se cambia a RGB

                # Verificar la cara cada 3 segundos
                if time.time() - self.last_verification_time >= 1:
                    # Encontrar todas las caras en el frame
                    self.face_locations = face_recognition.face_locations(rgb_small_frame)
                    self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                    self.face_names = []
                    for face_encoding in self.face_encodings:
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                        name = 'Desconocido'
                        confidence = 'Desconocido'

                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        best_match_i = np.argmin(face_distances)

                        if matches[best_match_i]:
                            name = self.known_face_names[best_match_i]
                            confidence = face_confidence(face_distances[best_match_i])

                        self.face_names.append(f'{name} ({confidence})')

                        # Capturar una foto cuando se detecta una cara desconocida
                        if name == 'Desconocido' and confidence == 'Desconocido' and not unknown_detected:
                            if unknown_start_time is None:
                                unknown_start_time = time.time()
                            elif time.time() - unknown_start_time >= 5:  # Esperar al menos 5 segundos antes de dar la opción de guardar la foto
                                cv2.imshow('Face Recognition', frame)
                                cv2.waitKey(500)  # Esperar medio segundo para estabilizar la imagen
                                ret, frame = video_capture.read()
                                self.save_unknown_face(frame)
                                unknown_detected = True
                                print("Foto guardada como 'desconocido.jpg'")

                        # Reiniciar el contador de tiempo cuando se detecta una cara conocida
                        if name != 'Desconocido':
                            unknown_start_time = None
                            unknown_detected = False

                    self.last_verification_time = time.time()

            self.process_current_frame = not self.process_current_frame
            # Mostrar los resultados
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Escalar las ubicaciones de las caras debido a que el marco en el que se detectó se redujo a 1/4 de tamaño
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Crear el marco con el nombre
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name.split(".")[0], (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            # Mostrar la imagen resultante
            cv2.imshow('Face Recognition', frame)

            # Presionar 'q' en el teclado para salir
            if cv2.waitKey(1) == ord('q'):
                break

        # Liberar la cámara
        video_capture.release()
        cv2.destroyAllWindows()

    @staticmethod
    def save_unknown_face(frame):
        name = input("Nombre de la persona desconocida: ")
        cv2.imwrite(f'faces/{name}.jpg', frame)
        print(f"Foto guardada como '{name}.jpg'")


if __name__ == '__main__':
    fr = FaceRecon()
    fr.run_recon()
