import cv2
import os

# Load Haar Cascade Classifier for Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Check if the cascade classifier loaded successfully
if face_cascade.empty():
    print("❌ Error: Could not load Haar Cascade. Ensure OpenCV is installed correctly.")
    exit()

def detect_faces_in_image(image_path):
    if not os.path.exists(image_path):
        print("❌ Error: The file does not exist. Please check the path.")
        return

    image = cv2.imread(image_path)
    if image is None:
        print("❌ Error: Could not read the image. Unsupported format?")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize image to improve face detection
    scale_percent = 100  # You can reduce this if detecting smaller faces
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    gray = cv2.resize(gray, (width, height))

    # Detect faces with optimized parameters
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display detection results
    if len(faces) == 0:
        print("ℹ️ No faces detected. Try a clearer image or adjust parameters.")
    else:
        print(f"✅ Detected {len(faces)} face(s).")

    cv2.imshow("Detected Faces", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main Program
image_path = input("📁 Enter full path to the image (without quotes): ").strip()
detect_faces_in_image(image_path)
