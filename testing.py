from google.cloud import vision
import io

def analyze_image(image_content):
    client = vision.ImageAnnotatorClient.from_service_account_file('color-insights-8e8f09462bd4.json')
    image = vision.Image(content=image_content)
    
    # Detect face features
    face_response = client.face_detection(image=image)
    faces = face_response.face_annotations
    
    # Detect labels
    label_response = client.label_detection(image=image)
    labels = label_response.label_annotations
    
    # Extract skin tone and facial features
    face_data = {}
    if faces:
        face = faces[0]  # Analyze the first face detected
        face_data = {
            'joy': face.joy_likelihood,
            'skin_tone': 'light' if face.detection_confidence > 0.8 else 'medium',  # Simplified detection
            'facial_features': {
                'eyes': face.landmarks_2d[0].position,
                'nose': face.landmarks_2d[1].position,
                'mouth': face.landmarks_2d[3].position
            }
        }
    
    return face_data, [{'description': label.description, 'score': label.score} for label in labels]

image_path = 'woman.jpg'

with io.open(image_path, 'rb') as image_file:
    content = image_file.read()
    
image = vision.Image(content=content)

response = client.label_detection(image=image)
labels = response.label_annotations

for label in labels:
    print(label.description, label.score)