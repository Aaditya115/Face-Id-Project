# Face Recognition System with DeepFace

## 📑 Summary
This project implements a **Face Recognition System** that captures frames from a webcam and compares detected faces with a reference image using **DeepFace** and **MTCNN**. The system detects faces from webcam frames, extracts embeddings, and calculates the **Cosine Similarity** between the captured face and the reference image to determine if the face matches. The result is shown with a label indicating the similarity percentage.

## 📚 Overview
This system uses the **MTCNN** for face detection and **DeepFace** for extracting face embeddings. The captured webcam frames are processed, and the system calculates the cosine similarity between the embeddings of the captured face and the reference face. If the similarity is above a threshold (85%), the face is labeled as valid.

### Features:
- **Webcam Integration**: Captures 3 frames from the webcam.
- **Face Detection**: Uses **MTCNN** to detect faces in each frame.
- **Face Recognition**: Compares detected faces with the reference image using **DeepFace**.
- **Similarity Calculation**: Uses **Cosine Similarity** to evaluate the match between faces.
- **Results Display**: Labels the frames with similarity percentage and validation status.
