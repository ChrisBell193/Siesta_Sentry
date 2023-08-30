import streamlit as st
import torch
import numpy as np
import cv2
import time
from pytube import YouTube
from PIL import Image
import os
import pygame
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
# Charger le modèle YOLOv5
if 'model' not in st.session_state:
    st.session_state['model'] = YOLO('/home/giovanni/code/GiovanniJolard/test_yolo_streamlit/runs_10/detect/train5/weights/best.pt')
model = st.session_state['model']
# A installer ffprobe alert_sound.wav
#211C3E
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# model = YOLO('/home/giovanni/code/GiovanniJolard/test_yolo_streamlit/runs_6/detect/yolov8s/weights/best.pt')
# Initialization
# model = YOLO('/home/giovanni/code/GiovanniJolard/test_yolo_streamlit/runs_6/detect/yolov8s/weights/best.pt')
# Configuration de la page Streamlit
st.set_page_config(
    page_title="Siesta Sentry",
    page_icon=":male-detective:",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Charger l'image pour le titre
title_image = Image.open("/mnt/c/Users/ruiz-/OneDrive/Images/SIESTA8LOGO.png")
# Afficher l'image du titre
st.image(title_image, use_column_width=True)
# Charger le son d'alerte
alert_sound_path = 'alert.wav'
# Initialiser pygame mixer
pygame.mixer.init()
# Champ de saisie pour l'URL de la vidéo YouTube ou le chemin du fichier
st.write("\n\n")
video_input_option = st.radio("Choisissez une option :", ("URL de la vidéo YouTube", "Chemin du fichier vidéo"))
if video_input_option == "URL de la vidéo YouTube":
    video_url = st.text_input("Entrez l'URL de la vidéo YouTube :")
    video_path = False
else:
    video_url = False
    video_path = st.file_uploader("Téléchargez une vidéo", type=["mp4"])
# Vérifier si l'utilisateur a soumis une URL de vidéo ou téléchargé une vidéo
if st.button("Démarrer la Détection") and (video_url or video_path):
    try:
        if video_url:
            # Télécharger la vidéo YouTube
            yt = YouTube(video_url)
            video_stream = yt.streams.filter(file_extension="mp4").first()
            video_path = video_stream.download()
        elif video_path is not None:
            # Enregistrer temporairement le fichier téléchargé
            with open(video_path.name, "wb") as f:
                f.write(video_path.read())
            video_path = video_path.name
        # Ouvrir la vidéo en streaming
        cap = cv2.VideoCapture(str(video_path))
        video_elem = st.empty()
        running = True
        stop_button = st.button("Arrêter la Détection")
        drowsy_counter = 0
        threshold_frames = 5
        last_alert_time = 0
        while running:
            ret, frame = cap.read()
            if not ret or stop_button:
                running = False
                break
            # Effectuer la détection
            results = model(frame)
            for result in results:
                names_dict = result.names
                if 'drowsy' in names_dict.values():
                    drowsy_counter += 1
                    if drowsy_counter >= threshold_frames:
                        current_time = time.time()
                        if current_time - last_alert_time >= 20:
                            # Jouer le son d'alerte
                            pygame.mixer.Sound(alert_sound_path).play()
                            last_alert_time = current_time
                            st.info("Endormi détecté !")
                else:
                    drowsy_counter = 0
                # Utilisez l'attribut result.orig_img pour afficher l'image originale
                detection_image = np.array(result.orig_img)
                # Dessinez les boîtes de détection sur l'image (si result.boxes contient les coordonnées)
                if result.boxes is not None:
                    for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                        x1, y1, x2, y2 = box.tolist()
                        class_name = names_dict[int(cls)]
                        label = f"{class_name} ({conf:.2f})"
                        # Dessiner la boîte de détection et le label
                        cv2.rectangle(detection_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(detection_image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Afficher l'image de détection
                video_elem.image(detection_image, channels="BGR", use_column_width=True)
                cv2.waitKey(1)
        # Afficher un message lorsque la détection est arrêtée
        st.write("Détection arrêtée")
    except Exception as e:
        # Afficher un message d'erreur en cas de problème
        st.error("Une erreur s'est produite lors du traitement de la vidéo.")
        st.error(str(e))
