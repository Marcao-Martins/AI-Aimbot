import cv2
import numpy as np
from ultralytics import YOLO
import os

class YOLODetection:
    def __init__(self, weights_path=None, conf_threshold=0.25, valid_classes=None):
        """
        Inicializa a detecção YOLO com um modelo específico.
        
        :param weights_path: Caminho para os pesos do modelo treinado (.pt)
        :param conf_threshold: Limite de confiança para detecções (0-1)
        :param valid_classes: Lista de classes para detectar. Se None, detecta todas as classes
        """
        try:
            self.model = YOLO(weights_path)
            self.conf_threshold = conf_threshold
            
            # Se valid_classes não for especificado, usa todas as classes do modelo
            self.valid_classes = valid_classes
            
            print(f"Modelo carregado de: {weights_path}")
            if self.valid_classes:
                print(f"Classes válidas: {self.valid_classes}")
            else:
                print("Detectando todas as classes disponíveis")
            
        except Exception as e:
            print(f"Erro ao carregar o modelo: {str(e)}")
            self.model = None

    def detect_objects(self, frame):
        """
        Detecta objetos em um frame usando o modelo YOLO.
        
        :param frame: Frame de entrada (NumPy array)
        :return: Lista de detecções, cada uma com coordenadas de bounding box e confiança
        """
        if self.model is None:
            print("Modelo não foi carregado corretamente")
            return []

        try:
            # Realiza a detecção
            results = self.model.predict(frame, conf=self.conf_threshold)

            # Processa os resultados
            detections = []
            for result in results:
                for box in result.boxes:
                    # Obtém o nome da classe
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    
                    # Se valid_classes não foi especificado ou a classe está na lista
                    if not self.valid_classes or class_name in self.valid_classes:
                        # Acessa as coordenadas da bounding box
                        x1, y1, x2, y2 = box.xyxy[0]
                        confidence = box.conf[0]
                        
                        detections.append({
                            "box": (int(x1), int(y1), int(x2), int(y2)),
                            "confidence": float(confidence),
                            "class_id": class_id,
                            "class_name": class_name
                        })

            return detections
        
        except Exception as e:
            print(f"Erro durante a detecção: {str(e)}")
            return []

    def draw_detections(self, frame, detections):
        """
        Desenha as detecções no frame.
        
        :param frame: Frame de entrada (NumPy array)
        :param detections: Lista de detecções
        :return: Frame com detecções desenhadas
        """
        frame_copy = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection["box"]
            confidence = detection["confidence"]
            class_name = detection["class_name"]
            
            # Desenha a bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepara o texto com classe e confiança
            label = f"{class_name}: {confidence:.2f}"
            
            # Adiciona o texto acima da bounding box
            cv2.putText(frame_copy, 
                       label, 
                       (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, 
                       (0, 255, 0), 
                       2)

        return frame_copy

    @staticmethod
    def list_available_weights(weights_dir='runs/train'):
        """
        Lista todos os pesos disponíveis no diretório de treinamento.
        
        :param weights_dir: Diretório onde os pesos estão armazenados
        :return: Lista de caminhos para arquivos de peso (.pt)
        """
        weights_files = []
        try:
            for root, dirs, files in os.walk(weights_dir):
                for file in files:
                    if file.endswith('.pt'):
                        weights_files.append(os.path.join(root, file))
            return weights_files
        except Exception as e:
            print(f"Erro ao listar pesos: {str(e)}")
            return []