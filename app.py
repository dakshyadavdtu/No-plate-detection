# streamlit run "/Users/dakshyadav/HTML TUTORIAL/python/no_plate_project/main/app.py"

import os
import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import easyocr
import pandas as pd
from util import set_background, write_csv
import uuid
from streamlit_webrtc import webrtc_streamer
import av
import tempfile
import time

set_background("/Users/dakshyadav/HTML TUTORIAL/python/no_plate_project/uploads/test_background.jpg")
folder_path = "/Users/dakshyadav/Desktop/License Detector"
LICENSE_MODEL_DETECTION_DIR ="/Users/dakshyadav/HTML TUTORIAL/python/no_plate_project/uploads/license_plate_detector.pt"
COCO_MODEL_DIR = "/Users/dakshyadav/HTML TUTORIAL/python/no_plate_project/uploads/yolov8n.pt"

reader = easyocr.Reader(['en'], gpu=True)

vehicles = [2]

header = st.container()
body = st.container()

coco_model = YOLO(COCO_MODEL_DIR).to("cpu")
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR).to("cpu")

threshold = 0.15

######## check for gpu #########

coco_model = YOLO(COCO_MODEL_DIR)  # Model initialization
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)

# Optionally check if YOLO is using GPU
print(f"YOLO model is running on GPU: {coco_model.device}")

###########################

# Ensure CSV directory exists
csv_dir = "License-Plate-Detection-with-YoloV8-and-EasyOCR/csv_detections"
os.makedirs(csv_dir, exist_ok=True)
csv_path = f"{csv_dir}/detection_results.csv"

# Initialize session state for tracking unique plates
if 'detected_plates' not in st.session_state:
    st.session_state['detected_plates'] = set()

state = "Uploader"

if "state" not in st.session_state:
    st.session_state["state"] = "Uploader"

class VideoProcessor:
    def __init__(self):
        self.detected_plates = set()

    # Add this after the existing state declaration
    def change_state_rtsp():
        st.session_state["state"] = "RTSP"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_to_an = img.copy()
        img_to_an = cv2.cvtColor(img_to_an, cv2.COLOR_RGB2BGR)
        license_detections = license_plate_detector(img_to_an)[0]

        if len(license_detections.boxes.cls.tolist()) != 0:
            for license_plate in license_detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

                license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]

                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, img)

                cv2.rectangle(img, (int(x1) - 40, int(y1) - 40), (int(x2) + 40, int(y1)), (255, 255, 255), cv2.FILLED)
                cv2.putText(img,
                            str(license_plate_text),
                            (int((int(x1) + int(x2)) / 2) - 70, int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 0),
                            3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def read_license_plate(license_plate_crop, img):
    scores = 0
    detections = reader.readtext(license_plate_crop)

    width = img.shape[1]
    height = img.shape[0]

    if detections == []:
        return None, None

    rectangle_size = license_plate_crop.shape[0]*license_plate_crop.shape[1]

    plate = []

    for result in detections:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if length*height / rectangle_size > 0.17:
            bbox, text, score = result
            text = result[1]
            text = text.upper()
            scores += score
            plate.append(text)

    if len(plate) != 0:
        return " ".join(plate), scores/len(plate)
    else:
        return None, None


def initialize_csv():
    """Initialize CSV file with headers"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Create a new CSV file with headers
        with open(csv_path, 'w') as f:
            f.write("vehicle_id,vehicle_bbox,vehicle_score,license_plate_bbox,license_plate_text,license_plate_bbox_score,license_plate_text_score\n")
    except Exception as e:
        print(f"Error initializing CSV: {str(e)}")


def process_video(video_file):
    try:
        # Initialize CSV file
        initialize_csv()
        
        # Reset detected plates
        st.session_state['detected_plates'] = set()
        
        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(video_file.read())
            temp_file_path = temp_file.name

        # Open the video file using the temporary file path
        cap = cv2.VideoCapture(temp_file_path)
        
        if not cap.isOpened():
            raise Exception("Could not open video file")
            
        results = {}
        license_id = 0
        frame_count = 0
        
        # Read and process frames from the video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            # Process every 5th frame to improve performance
            if frame_count % 5 != 0:
                continue
                
            # Process the frame
            img = frame.copy()
            object_detections = coco_model(img)[0]
            license_detections = license_plate_detector(img)[0]
            
            # Get car detections
            car_bbox = [0, 0, 0, 0]
            car_score = 0
            
            if len(object_detections.boxes.cls.tolist()) != 0:
                for detection in object_detections.boxes.data.tolist():
                    xcar1, ycar1, xcar2, ycar2, score, class_id = detection
                    if int(class_id) in vehicles:
                        car_bbox = [xcar1, ycar1, xcar2, ycar2]
                        car_score = score
                        break  # Just use the first car detected
            
            # Process license plates
            if len(license_detections.boxes.cls.tolist()) != 0:
                for license_plate in license_detections.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = license_plate
                    
                    license_plate_crop = img[int(y1):int(y2), int(x1):int(x2), :]
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, img)
                    
                    # Only add non-None license plates that haven't been seen before
                    if license_plate_text and license_plate_text not in st.session_state['detected_plates']:
                        st.session_state['detected_plates'].add(license_plate_text)
                        
                        # Save the license plate image
                        img_name = '{}.jpg'.format(uuid.uuid1())
                        try:
                            cv2.imwrite(os.path.join(folder_path, img_name), license_plate_crop)
                        except Exception as e:
                            print(f"Error saving license plate image: {str(e)}")
                        
                        # Add to results
                        results[license_id] = {
                            'car': {'bbox': car_bbox, 'car_score': car_score},
                            'license_plate': {
                                'bbox': [x1, y1, x2, y2],
                                'text': license_plate_text,
                                'bbox_score': score,
                                'text_score': license_plate_text_score
                            }
                        }
                        license_id += 1
        
        # Release the video capture object
        cap.release()
        
        # Remove the temporary file after processing
        os.remove(temp_file_path)
        
        # Write results to CSV if we have any
        if results:
            write_csv(results, csv_path)
            
        return results
    except Exception as e:
        print(f"Error in process_video: {str(e)}")
        st.error(f"Error processing video: {str(e)}")
        return {}


def model_prediction(img):
    try:
        # Initialize CSV file
        initialize_csv()
        
        # Reset detected plates
        st.session_state['detected_plates'] = set()
        
        license_numbers = 0
        results = {}
        licenses_texts = []
        license_plate_crops_total = []
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        object_detections = coco_model(img)[0]
        license_detections = license_plate_detector(img)[0]

        # Initialize car detection variables
        xcar1, ycar1, xcar2, ycar2, car_score = 0, 0, 0, 0, 0

        # Get car detections
        if len(object_detections.boxes.cls.tolist()) != 0:
            for detection in object_detections.boxes.data.tolist():
                xcar1, ycar1, xcar2, ycar2, car_score, class_id = detection
                if int(class_id) in vehicles:
                    cv2.rectangle(img, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 0, 255), 3)
                    break  # Just use the first car detected

        # Process license plates
        if len(license_detections.boxes.cls.tolist()) != 0:
            for license_plate in license_detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                license_plate_crop = img[int(y1):int(y2), int(x1):int(x2), :]
                
                # Save the license plate image
                img_name = '{}.jpg'.format(uuid.uuid1())
                try:
                    cv2.imwrite(os.path.join(folder_path, img_name), license_plate_crop)
                except Exception as e:
                    print(f"Error saving license plate image: {str(e)}")

                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, img)
                
                # Only process valid license plates
                if license_plate_text:
                    licenses_texts.append(license_plate_text)
                    license_plate_crops_total.append(license_plate_crop)
                    
                    # Add text to image
                    cv2.rectangle(img, (int(x1) - 40, int(y1) - 40), (int(x2) + 40, int(y1)), (255, 255, 255), cv2.FILLED)
                    cv2.putText(img,
                                str(license_plate_text),
                                (int((int(x1) + int(x2)) / 2) - 70, int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 0),
                                3)
                    
                    # Only add to results if it's a unique plate
                    if license_plate_text not in st.session_state['detected_plates']:
                        st.session_state['detected_plates'].add(license_plate_text)
                        
                        results[license_numbers] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2], 'car_score': car_score},
                            'license_plate': {'bbox': [x1, y1, x2, y2], 'text': license_plate_text,
                                            'bbox_score': score, 'text_score': license_plate_text_score}
                        }
                        license_numbers += 1

            # Write results to CSV if we have any
            if results:
                write_csv(results, csv_path)

        img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return [img_wth_box, licenses_texts, license_plate_crops_total]
    except Exception as e:
        print(f"Error in model_prediction: {str(e)}")
        # Return original image and empty lists if an error occurs
        if isinstance(img, np.ndarray):
            try:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return [img_rgb, [], []]
            except:
                return [img, [], []]
        return [img, [], []]


def change_state_uploader():
    st.session_state["state"] = "Uploader"


def change_state_camera():
    st.session_state["state"] = "Camera"


def change_state_live():
    st.session_state["state"] = "Live"


def change_state_rtsp():
    st.session_state["state"] = "RTSP"


def download_csv():
    """Create a download button for the CSV file"""
    try:
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                csv_data = f.read()
                
            # Check if CSV has data beyond headers
            if len(csv_data.strip().split('\n')) > 1:
                st.download_button(
                    label="Download CSV Results",
                    data=csv_data,
                    file_name="license_plate_detections.csv",
                    mime="text/csv"
                )
            else:
                st.warning("CSV file exists but contains no data.")
        else:
            st.warning("No CSV file generated yet.")
    except Exception as e:
        st.error(f"Error preparing CSV download: {str(e)}")


with header:
    _, col1, _ = st.columns([0.2,1,0.1])
    col1.title("💥 License Car Plate Detection 🚗")

    _, col0, _ = st.columns([0.15,1,0.1])
    col0.image("/Users/dakshyadav/HTML TUTORIAL/python/no_plate_project/uploads/test_background.jpg", width=500)


    _, col4, _ = st.columns([0.1,1,0.2])
    col4.subheader("Computer Vision Detection with YoloV8 🧪")

    _, col, _ = st.columns([0.3,1,0.1])
    col.image("/Users/dakshyadav/HTML TUTORIAL/python/no_plate_project/uploads/plate_test.jpg")

    _, col5, _ = st.columns([0.05,1,0.1])

    st.write("The differents models detect the car and the license plate in a given image, then extracts the info about the license using EasyOCR, and crop and save the license plate as a Image, with a CSV file with all the data.")


with body:
    _, col1, _ = st.columns([0.1,1,0.2])
    col1.subheader("Check It-out the License Car Plate Detection Model 🔎!")

    _, colb1, colb2, colb3, colb4 = st.columns([0.2, 0.7, 0.6, 0.6, 1])

    if colb1.button("Upload an Image or Video ", on_click=change_state_uploader):
        pass
    elif colb2.button("Take a Photo", on_click=change_state_camera):
        pass
    elif colb3.button("Live Detection", on_click=change_state_live):
        pass
    elif colb4.button("RTSP Stream", on_click=change_state_rtsp):
        pass

    if st.session_state["state"] == "Uploader":
        # Reset detected plates when uploading a new file
        if st.button("Reset Detection History"):
            st.session_state['detected_plates'] = set()
            initialize_csv()
            st.success("Detection history reset. CSV file cleared.")
            
        img = st.file_uploader("Upload a Car Image or Video: ", type=["png", "jpg", "jpeg", "mp4"])

        if img is not None:
            try:
                if img.name.endswith('.mp4'):
                    # Process video
                    results = process_video(img)
                    
                    if results:
                        st.success(f"Video processed successfully! {len(results)} unique license plates detected.")
                        
                        # Display CSV data
                        try:
                            df = pd.read_csv(csv_path)
                            if not df.empty:
                                st.dataframe(df)
                                download_csv()
                            else:
                                st.warning("CSV file is empty. No license plates were detected.")
                        except Exception as e:
                            st.error(f"Error reading CSV: {str(e)}")
                    else:
                        st.warning("No license plates detected in the video or processing failed.")
                else:
                    # Process image
                    image = np.array(Image.open(img))
                    results = model_prediction(image)
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("If you uploaded a video file, make sure it's a valid MP4. If you uploaded an image, make sure it's a valid image format.")
        else:
            st.warning("Please upload an image or video file.")

    elif st.session_state["state"] == "Camera":
        img = st.camera_input("Take a Photo: ")

    elif st.session_state["state"] == "Live":
        webrtc_streamer(key="sample", video_processor_factory=VideoProcessor)
        img = None
        
        # Add CSV download button for live detection
        if os.path.exists(csv_path):
            download_csv()

    elif st.session_state["state"] == "RTSP":
        rtsp_url = st.text_input(
            "RTSP URL",
            value="rtsp://admin:Admin%40123@10.39.1.25:554/cam/realmonitor?channel=1&subtype=0"
        )
        
        if st.button("Start RTSP Stream"):
            # Initialize CSV file
            initialize_csv()
            
            # Reset detected plates
            st.session_state['detected_plates'] = set()
            
            try:
                cap = cv2.VideoCapture(rtsp_url)
                
                if not cap.isOpened():
                    st.error("Error: Could not open RTSP stream")
                else:
                    video_placeholder = st.empty()
                    
                    while True:
                        try:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            # Process frame using existing model_prediction function
                            results = model_prediction(frame)
                            
                            if len(results) == 3:
                                prediction, texts, license_plate_crop = results[0], results[1], results[2]
                                
                                # Display the processed frame
                                video_placeholder.image(prediction)
                                
                                # Display detected license plates
                                if texts:
                                    for i, text in enumerate(texts):
                                        if text:
                                            st.write(f"Detected License Plate {i+1}: {text}")
                            
                            # Add a small delay to prevent overwhelming the system
                            time.sleep(0.1)
                        except Exception as e:
                            st.error(f"Error processing RTSP frame: {str(e)}")
                            time.sleep(1)  # Wait a bit before trying again
                            continue
                    
                    cap.release()
            except Exception as e:
                st.error(f"Error with RTSP stream: {str(e)}")
        
        # Add CSV download button for RTSP detection
        if os.path.exists(csv_path):
            download_csv()

    _, col2, _ = st.columns([0.3,1,0.2])
    _, col5, _ = st.columns([0.8,1,0.2])

    if img is not None and not img.name.endswith('.mp4'):
        try:
            image = np.array(Image.open(img))
            col2.image(image, width=400)

            if col5.button("Apply Detection"):
                try:
                    results = model_prediction(image)

                    if len(results) == 3:
                        prediction, texts, license_plate_crop = results[0], results[1], results[2]

                        texts = [i for i in texts if i is not None]

                        if len(texts) == 1 and len(license_plate_crop):
                            _, col3, _ = st.columns([0.4,1,0.2])
                            col3.header("Detection Results ✅:")

                            _, col4, _ = st.columns([0.1,1,0.1])
                            col4.image(prediction)

                            _, col9, _ = st.columns([0.4,1,0.2])
                            col9.header("License Cropped ✅:")

                            _, col10, _ = st.columns([0.3,1,0.1])
                            col10.image(license_plate_crop[0], width=350)

                            _, col11, _ = st.columns([0.45,1,0.55])
                            col11.success(f"License Number: {texts[0]}")

                            try:
                                df = pd.read_csv(csv_path)
                                if len(df) > 0:
                                    st.dataframe(df)
                                    download_csv()
                                else:
                                    st.warning("CSV file is empty. No license plates were detected.")
                            except Exception as e:
                                st.error(f"Error reading CSV: {str(e)}")
                        elif len(texts) > 1 and len(license_plate_crop) > 1:
                            _, col3, _ = st.columns([0.4,1,0.2])
                            col3.header("Detection Results ✅:")

                            _, col4, _ = st.columns([0.1,1,0.1])
                            col4.image(prediction)

                            _, col9, _ = st.columns([0.4,1,0.2])
                            col9.header("License Cropped ✅:")

                            _, col10, _ = st.columns([0.3,1,0.1])
                            _, col11, _ = st.columns([0.45,1,0.55])

                            for i in range(0, len(license_plate_crop)):
                                col10.image(license_plate_crop[i], width=350)
                                col11.success(f"License Number {i}: {texts[i]}")

                            try:
                                df = pd.read_csv(csv_path)
                                if len(df) > 0:
                                    st.dataframe(df)
                                    download_csv()
                                else:
                                    st.warning("CSV file is empty. No license plates were detected.")
                            except Exception as e:
                                st.error(f"Error reading CSV: {str(e)}")
                        else:
                            st.warning("No license plates detected in the image.")
                    else:
                        prediction = results[0]
                        _, col3, _ = st.columns([0.4,1,0.2])
                        col3.header("Detection Results ✅:")

                        _, col4, _ = st.columns([0.3,1,0.1])
                        col4.image(prediction)
                except Exception as e:
                    st.error(f"Error during detection: {str(e)}")
        except Exception as e:
            st.error(f"Error opening image: {str(e)}")
            if img.name.endswith('.mp4'):
                st.info("This appears to be a video file. Please use the 'Upload an Image or Video' option and ensure the file is properly recognized as a video.")



