from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

# Load the pre-trained model
model_json_file = "Emotion-model.json"
model_weights_file = "FacialExpression_weights.hdf5"

with open(model_json_file, "r") as json_file:
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
model.load_weights(model_weights_file)

# Load the Haar Cascade classifier for face detection
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

class VideoProcessor:
	def recv(self, frame):
		frm = frame.to_ndarray(format="bgr24")

		faces = cascade.detectMultiScale(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY), 1.1, 3)

		for x,y,w,h in faces:
			cv2.rectangle(frm, (x,y), (x+w, y+h), (0,255,0), 3)

		return av.VideoFrame.from_ndarray(frm, format='bgr24')

webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
				rtc_configuration=RTCConfiguration(
					{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
					)
	)

