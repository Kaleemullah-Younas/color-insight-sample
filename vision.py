from google.cloud import vision
from PIL import Image
import io
import numpy as np
import colorsys

def analyze_image(image_content):
    # —— Vision API calls ——
    client = vision.ImageAnnotatorClient.from_service_account_file('color-insights-5994bd240a4d.json')
    image = vision.Image(content=image_content)
    face_resp  = client.face_detection(image=image)
    label_resp = client.label_detection(image=image)
    faces  = face_resp.face_annotations
    labels = label_resp.label_annotations

    # —— Decode image for pixel sampling ——
    pil_img = Image.open(io.BytesIO(image_content)).convert('RGB')
    img_arr  = np.array(pil_img)

    def rgb_to_hue(rgb):
        r, g, b = [v/255.0 for v in rgb]
        h, _, _ = colorsys.rgb_to_hsv(r, g, b)
        return h * 360

    def sample_patch(pt, w=5):
        if not pt: return None
        x, y = int(pt.x), int(pt.y)
        y1, y2 = max(y-w, 0), min(y+w, img_arr.shape[0])
        x1, x2 = max(x-w, 0), min(x+w, img_arr.shape[1])
        patch = img_arr[y1:y2, x1:x2]
        if patch.size == 0: return None
        return patch.reshape(-1,3).mean(axis=0)

    feature_list = []
    for face in faces:
        # 1) simple emotions & orientation
        feats = {
            'joy'      : face.joy_likelihood,
            'sorrow'   : face.sorrow_likelihood,
            'anger'    : face.anger_likelihood,
            'surprise' : face.surprise_likelihood,
            'roll_angle': face.roll_angle,   # tilt
            'pan_angle' : face.pan_angle,    # yaw
            'tilt_angle': face.tilt_angle,   # pitch
        }

        # 2) collect landmark positions
        lm = {l.type: l.position for l in face.landmarks}
        # key landmarks
        pts = {
            'left_eye'           : lm.get(vision.FaceAnnotation.Landmark.Type.LEFT_EYE),
            'right_eye'          : lm.get(vision.FaceAnnotation.Landmark.Type.RIGHT_EYE),
            'nose_tip'           : lm.get(vision.FaceAnnotation.Landmark.Type.NOSE_TIP),
            'nose_bottom_center' : lm.get(vision.FaceAnnotation.Landmark.Type.NOSE_BOTTOM_CENTER),
            'nose_bottom_right'  : lm.get(vision.FaceAnnotation.Landmark.Type.NOSE_BOTTOM_RIGHT),
            'nose_bottom_left'   : lm.get(vision.FaceAnnotation.Landmark.Type.NOSE_BOTTOM_LEFT),
            'mouth_center'       : lm.get(vision.FaceAnnotation.Landmark.Type.MOUTH_CENTER),
            'upper_lip'          : lm.get(vision.FaceAnnotation.Landmark.Type.UPPER_LIP),
            'lower_lip'          : lm.get(vision.FaceAnnotation.Landmark.Type.LOWER_LIP),
            'mouth_left'         : lm.get(vision.FaceAnnotation.Landmark.Type.MOUTH_LEFT),
            'mouth_right'        : lm.get(vision.FaceAnnotation.Landmark.Type.MOUTH_RIGHT),
            'left_eyebrow_mid'   : lm.get(vision.FaceAnnotation.Landmark.Type.LEFT_EYEBROW_UPPER_MIDPOINT),
            'right_eyebrow_mid'  : lm.get(vision.FaceAnnotation.Landmark.Type.RIGHT_EYEBROW_UPPER_MIDPOINT),
            'left_cheek_center'  : lm.get(vision.FaceAnnotation.Landmark.Type.LEFT_CHEEK_CENTER),
            'right_cheek_center' : lm.get(vision.FaceAnnotation.Landmark.Type.RIGHT_CHEEK_CENTER),
            'chin_gnathion'      : lm.get(vision.FaceAnnotation.Landmark.Type.CHIN_GNATHION),
            'chin_left'          : lm.get(vision.FaceAnnotation.Landmark.Type.CHIN_LEFT_GONION),
            'chin_right'         : lm.get(vision.FaceAnnotation.Landmark.Type.CHIN_RIGHT_GONION),
            'forehead_glabella'  : lm.get(vision.FaceAnnotation.Landmark.Type.FOREHEAD_GLABELLA),
        }   
        # add raw coords
        for name, pt in pts.items():
            feats[name] = (pt.x, pt.y) if pt else None

        # 3) geometric ratios (normalized to bbox width)
        verts = face.bounding_poly.vertices
        xs = [v.x for v in verts]; ys = [v.y for v in verts]
        x_min, x_max = max(min(xs), 0), min(max(xs), img_arr.shape[1])
        y_min, y_max = max(min(ys), 0), min(max(ys), img_arr.shape[0])
        bbox_w = x_max - x_min

        # inter-eye distance
        if pts['left_eye'] and pts['right_eye']:
            eye_dist = np.hypot(
                pts['left_eye'].x - pts['right_eye'].x,
                pts['left_eye'].y - pts['right_eye'].y
            )
            feats['inter_eye_dist_norm'] = eye_dist / bbox_w

        # nose-to-mouth
        if pts['nose_tip'] and pts['mouth_center']:
            nm_dist = np.hypot(
                pts['nose_tip'].x - pts['mouth_center'].x,
                pts['nose_tip'].y - pts['mouth_center'].y
            )
            feats['nose_to_mouth_norm'] = nm_dist / bbox_w

        # 4) color sampling
        # skin tone
        face_crop = img_arr[y_min:y_max, x_min:x_max]
        avg_skin = face_crop.reshape(-1,3).mean(axis=0)
        feats['skin_rgb'] = {'r':float(avg_skin[0]), 'g':float(avg_skin[1]), 'b':float(avg_skin[2])}
        feats['skin_hue'] = float(rgb_to_hue(avg_skin))

        # hair color (above forehead)
        height = y_max - y_min
        h1 = max(int(y_min - height*0.3), 0)
        h2 = y_min + int(height*0.1)
        hair_crop = img_arr[h1:h2, x_min:x_max]
        if hair_crop.size:
            avg_hair = hair_crop.reshape(-1,3).mean(axis=0)
        else:
            avg_hair = avg_skin
        feats['hair_rgb'] = {'r':float(avg_hair[0]), 'g':float(avg_hair[1]), 'b':float(avg_hair[2])}
        feats['hair_hue'] = float(rgb_to_hue(avg_hair))

        # eye color
        left_rgb  = sample_patch(pts['left_eye'])
        right_rgb = sample_patch(pts['right_eye'])
        if left_rgb is not None or right_rgb is not None:
            combined = ((left_rgb + right_rgb)/2) if (left_rgb is not None and right_rgb is not None) else (left_rgb or right_rgb)
            feats['eye_rgb'] = {'r':float(combined[0]), 'g':float(combined[1]), 'b':float(combined[2])}
            feats['eye_hue'] = float(rgb_to_hue(combined))

        feature_list.append(feats)

    # return everything plus labels
    return feature_list, [
        {'description': l.description, 'score': l.score}
        for l in labels
    ]