from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import streamlit as st
import cv2
import numpy as np



# Load the pre-trained model
model_json_file = "Emotion-model.json"
model_weights_file = "FacialExpression_weights.hdf5"

)

# Load the Haar Cascade classifier for face detection
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# Define the list of emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]   



st.set_page_config(page_title="Facial Expression Recognition", page_icon=":mango:")
st.title("Facial Expression Recognition with Streamlit")

with st.sidebar:
    st.header("Facial Expression Recognition")
    st.title("Facial Expression Recognition Prediction")
    st.subheader("Facial expression recognition enables more natural and intuitive interactions between humans and computer systems, enhancing user experience and engagement.")

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
