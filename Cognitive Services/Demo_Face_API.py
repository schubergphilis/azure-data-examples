#%%
import imageio as iio
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import os
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
import streamlit as st

# Useful links:
# https://pbpython.com/python-face-detect.html
# https://sebastianwallkoetter.wordpress.com/2021/03/07/webcam-in-python/


def grab_frame_webcam():
    """
    Returns:
       stream: byte stream generated from single camera frame
       frame: array, rbg representation of the frame
    """
    camera = iio.get_reader("<video0>")
    frame = camera.get_data(0)
    camera.close()

    # The API requires a data stream, so we convert.
    # buf will be the encoded image
    ret, buf = cv2.imencode(".jpg", frame)

    # stream-ify the buffer
    stream = io.BytesIO(buf)

    return stream, frame


def drawFaceRectangles(frame, detected_faces):
    """
    Draw the image including rectangles over the detected fases

    Args:
        frame: array, rbg representation of the frame
        detected_faces list: list of all the detected faces

    Returns:
        fig: matplotlib figure object

    """
    fig, ax = plt.subplots()
    ax.imshow(frame)
    for face in detected_faces:
        fr = face.face_rectangle
        fa = face.face_attributes
        emotion = face.face_attributes.emotion.__dict__
        emotion.pop("additional_properties")
        dominant_emotion = max(emotion, key=emotion.get)

        origin = (fr.left, fr.top)
        p = patches.Rectangle(origin, fr.width, fr.height, fill=False, linewidth=2, color="b")
        ax.axes.add_patch(p)
        ax.text(
            origin[0],
            origin[1],
            "%s, %d, %s" % (fa.gender.capitalize(), fa.age, dominant_emotion),
            fontsize=8,
            weight="bold",
            va="bottom",
        )
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    return fig


# Key and end-point
KEY = os.getenv("AZURE_FACE_API_KEY")
ENDPOINT = os.getenv("AZURE_FACE_API_ENDPOINT")

# Create an authenticated FaceClient.
face_client = FaceClient(
    ENDPOINT,
    CognitiveServicesCredentials(KEY),
)

st.header("Demo of Azure Cognitive Services Face API")
st.sidebar.subheader("Get started!")
if st.sidebar.button("Take a picture"):

    # Single frame from webcam
    stream, frame = grab_frame_webcam()

    # Call face api on created stream
    detected_faces = face_client.face.detect_with_stream(
        stream,
        return_face_landmarks=True,
        return_face_attributes=["age", "gender", "hair", "emotion"],
        recognition_model="recognition_04",
        detection_model="detection_01",
        return_recognition_model=True,
    )

    if not detected_faces:
        raise Exception("No face detected from image {}".format(frame))

    fig = drawFaceRectangles(frame, detected_faces)

    st.write(fig)

# Scratch pad
# How to access more face attributes:
# my_face = detected_faces[0]
# age = my_face.face_attributes.age
# gender = my_face.face_attributes.gender
# mask = my_face.face_attributes.mask
# emotion = my_face.face_attributes.emotion.__dict__
# emotion.pop("additional_properties")