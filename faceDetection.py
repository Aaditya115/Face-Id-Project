import cv2
import numpy as np
from deepface import DeepFace
from mtcnn import MTCNN
from numpy.linalg import norm

# Initialize MTCNN face detector
detector = MTCNN()

# Load refrence image and create its embedding (ensure this is correct)
ref_image = cv2.imread("img.jpg")  # Replace with the path to your image
ref_image_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

# Get Aaditya's embedding using DeepFace (Ensure this works)
ref_embedding = DeepFace.represent(ref_image_rgb, model_name="Facenet")[0]["embedding"]

# Function to calculate Cosine Similarity
def cosine_similarity(embedding1, embedding2):
    # Cosine similarity = dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
    return np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))

# Start the webcam
video_capture = cv2.VideoCapture(0)

# Check if the webcam opened correctly
if not video_capture.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Loop to capture 3 frames
frames_captured = 0
captured_frames = []

while frames_captured < 3:
    ret, frame = video_capture.read()

    # Check if frame is read correctly
    if not ret:
        print("Error: Failed to capture image!")
        break

    # Show the frame in a window for debugging (optional)
    cv2.imshow("Webcam Feed", frame)

    # Add the frame to the captured frames list
    captured_frames.append(frame)
    frames_captured += 1

    # Wait for 1 millisecond and break after pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam after capturing 3 frames
video_capture.release()

if frames_captured != 3:
    print("Error: Failed to capture 3 frames!")
else:
    print("3 frames captured successfully.")

    # Process the captured frames
    for i, frame in enumerate(captured_frames):
        # Detect faces in the captured frame using MTCNN
        results = detector.detect_faces(frame)

        if not results:
            print(f"No faces detected in frame {i + 1}!")
        else:
            for result in results:
                # Get bounding box coordinates and draw a rectangle
                x, y, w, h = result['box']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Extract the detected face region
                face = frame[y:y + h, x:x + w]

                # Check if face is detected
                if face.size != 0:
                    # Convert to RGB for DeepFace processing
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                    try:
                        # Get embedding for the detected face using DeepFace (CNN-based recognition)
                        detected_embedding = DeepFace.represent(face_rgb, model_name="Facenet")[0]["embedding"]

                        # Calculate cosine similarity between Aaditya's embedding and the detected face's embedding
                        similarity = cosine_similarity(np.array(aaditya_embedding), np.array(detected_embedding))

                        # Convert cosine similarity to percentage similarity
                        similarity_percentage = max(0, min(similarity * 100, 100))

                        # Debug: print the similarity for inspection
                        print(f"Cosine Similarity for frame {i + 1}: {similarity}")
                        print(f"Similarity for frame {i + 1}: {similarity_percentage}%")

                        # Check validity based on similarity threshold (85%)
                        if similarity_percentage > 85:
                            print(f"Valid for frame {i + 1}")  # Similarity higher than 85% means "Valid"
                            name = f"ref ({similarity_percentage:.2f}%)"
                        else:
                            print(f"Invalid for frame {i + 1}")  # Similarity lower than 85% means "Invalid"
                            name = f"Unknown ({similarity_percentage:.2f}%)"
                    except ValueError:
                        print(f"Error in face recognition for frame {i + 1}.")
                        name = "Unknown"  # If face detection or recognition fails
                else:
                    name = "No Face Detected"

                # Add label to the face
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow(f'Captured Frame {i + 1}', frame)

        # Wait for user to press any key before closing
        cv2.waitKey(0)

# Close any OpenCV windows
cv2.destroyAllWindows()
