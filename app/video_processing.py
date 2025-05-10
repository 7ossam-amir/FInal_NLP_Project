import cv2
import numpy as np
import time
from PIL import Image, ImageFont, ImageDraw
import glob
import os
import threading  
from .models import get_models
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
except ImportError:
    print("Please install arabic_reshaper and python-bidi")
    raise

# Video data storage
video_data = {
    'filename': None,
    'video_path': None,
    'selected_id': None,
    'track_history': [],
    'all_detected_ids': {},
    'object_descriptions': {},
    'object_positions': {},
    'occlusion_count': {},
    'max_occlusion_frames': 60,
    'continuous_tracking': False,
    'track_end_time': None,
    'track_start_time': None,
    'full_description': '',
    'tracking_active': False,
    'current_frame': 0,
    'total_frames': 0,
    'tracking_summary': [],
    'activity_timeline': []
}

try:
    font = ImageFont.load_default()
except IOError:
    print("Font file not found. Using default font")

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def get_video_info(video_path):
    if not video_path or not os.path.exists(video_path):
        return {'filename': 'No video loaded', 'fps': 0, 'frame_count': 0, 'duration': 0}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'filename': os.path.basename(video_path), 'fps': 0, 'frame_count': 0, 'duration': 0}
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return {'filename': os.path.basename(video_path), 'fps': fps, 'frame_count': frame_count, 'duration': duration}

def generate_description(frame, obj_id):
    models = get_models()
    try:
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = models['blip_processor'](pil_frame, return_tensors="pt")
        out_desc = models['blip_model'].generate(**inputs, max_length=50)
        description = models['blip_processor'].decode(out_desc[0], skip_special_tokens=True)
        inputs_trans = models['translation_tokenizer'](description, return_tensors="pt")
        translated = models['translation_model'].generate(**inputs_trans, max_length=100)
        arabic_description = models['translation_tokenizer'].decode(translated[0], skip_special_tokens=True)
        timestamp = time.time()
        return {
            'timestamp': timestamp,
            'english': description,
            'arabic': arabic_description
        }
    except Exception as e:
        print(f"Error generating description: {e}")
        return {
            'timestamp': time.time(),
            'english': "Error generating description",
            'arabic': "خطأ في توليد الوصف"
        }

def extract_object_from_frame(frame, box, padding=10):
    x1, y1, x2, y2 = [int(coord) for coord in box]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(frame.shape[1], x2 + padding)
    y2 = min(frame.shape[0], y2 + padding)
    object_img = frame[y1:y2, x1:x2]
    return object_img if object_img.size > 0 else None

def iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection_area / float(box1_area + box2_area - intersection_area)

def handle_occlusion(track_id, detections, frame_time):
    current_ids = [det['id'] for det in detections]
    if track_id not in current_ids and track_id in video_data['object_positions'] and track_id in video_data['all_detected_ids']:
        if track_id not in video_data['occlusion_count']:
            video_data['occlusion_count'][track_id] = 0
        video_data['occlusion_count'][track_id] += 1
        if video_data['occlusion_count'][track_id] <= video_data['max_occlusion_frames']:
            last_position = video_data['object_positions'][track_id]
            last_class = video_data['all_detected_ids'][track_id]['class']
            detections.append({
                'id': track_id,
                'class': last_class,
                'box': last_position,
                'time': frame_time,
                'occluded': True
            })
            video_data['activity_timeline'].append({
                'time': frame_time,
                'event': 'occlusion',
                'duration': video_data['occlusion_count'][track_id]
            })
            return True
    else:
        if track_id in current_ids:
            if track_id in video_data['occlusion_count'] and video_data['occlusion_count'][track_id] > 0:
                video_data['activity_timeline'].append({
                    'time': frame_time,
                    'event': 'reappeared',
                    'duration': video_data['occlusion_count'][track_id]
                })
            video_data['occlusion_count'][track_id] = 0
    return False

def find_best_match(track_id, detections, iou_threshold=0.4):
    if track_id not in video_data['object_positions']:
        return None
    last_box = video_data['object_positions'][track_id]
    best_iou = 0
    best_detection = None
    for det in detections:
        if 'id' in det and det['id'] == track_id:
            continue
        current_iou = iou(last_box, det['box'])
        if current_iou > best_iou and current_iou > iou_threshold:
            best_iou = current_iou
            best_detection = det
    return best_detection

def pil_text_wrap(text, font, max_width):
    lines = []
    words = text.split(' ')
    if not words:
        return lines
    current_line = words[0]
    for word in words[1:]:
        line_with_word = current_line + ' ' + word
        if font.getlength(line_with_word) <= max_width:
            current_line = line_with_word
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return lines

def add_text_to_frame(frame, text, position, font, color=(255, 255, 255), is_arabic=False):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    if is_arabic:
        text = get_display(arabic_reshaper.reshape(text))
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def process_frame(video_path, frame_time, track_id=None, continuous_tracking=False, app_config=None):
    models = get_models()
    if not models['models_loaded']:
        try:
            models['model_queue'].get(timeout=0.1)
        except queue.Empty:
            return {
                'time': frame_time,
                'annotated_image': None,
                'detections': [],
                'frame_width': 0,
                'frame_height': 0,
                'loading': True,
                'models_ready': False
            }
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError("Invalid FPS value")
        frame_number = int(frame_time * fps)
        video_data['current_frame'] = frame_number
        video_data['total_frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()
        if not success:
            raise ValueError("Could not read frame")

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = models['detection_model'].track(
                rgb_frame,
                persist=True,
                tracker="botsort.yaml",
                conf=0.5,
                device='cuda'
            )
        except Exception as e:
            print(f"Detection model error: {e}")
            return {'error': 'Object detection failed', 'time': frame_time, 'models_ready': True}

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else None
        classes = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        names = results[0].names

        detections = []
        if ids is not None:
            for box, id, cls, conf in zip(boxes, ids, classes, confs):
                obj_id = int(id)
                obj_class = names[int(cls)]
                box_coords = [int(x) for x in box]
                detections.append({
                    'id': obj_id,
                    'class': obj_class,
                    'box': box_coords,
                    'time': frame_time,
                    'conf': float(conf)
                })

                if obj_id not in video_data['all_detected_ids']:
                    video_data['all_detected_ids'][obj_id] = {
                        'class': obj_class,
                        'first_seen': frame_time,
                        'appearances': 1,
                        'confidence': float(conf)
                    }
                else:
                    video_data['all_detected_ids'][obj_id]['appearances'] += 1
                    old_conf = video_data['all_detected_ids'][obj_id].get('confidence', 0)
                    appearances = video_data['all_detected_ids'][obj_id]['appearances']
                    video_data['all_detected_ids'][obj_id]['confidence'] = (old_conf * (appearances-1) + float(conf)) / appearances

                video_data['object_positions'][obj_id] = box_coords

        if track_id is not None:
            found = any(det['id'] == track_id for det in detections)
            if not found:
                occlusion_handled = handle_occlusion(track_id, detections, frame_time)
                if not occlusion_handled:
                    best_match = find_best_match(track_id, detections)
                    if best_match:
                        for det in detections:
                            if det['id'] == best_match['id']:
                                print(f"ID switching: {det['id']} -> {track_id} at frame {frame_number}")
                                det['id'] = track_id
                                video_data['object_positions'][track_id] = det['box']
                                video_data['activity_timeline'].append({
                                    'time': frame_time,
                                    'event': 'id_switch',
                                    'from_id': best_match['id'],
                                    'to_id': track_id
                                })
                                break

        if track_id is not None:
            tracked_det = next((det for det in detections if det['id'] == track_id), None)
            if tracked_det:
                object_img = extract_object_from_frame(frame, tracked_det['box'])
                should_generate = True
                if track_id in video_data['object_descriptions'] and video_data['object_descriptions'][track_id]:
                    last_desc = video_data['object_descriptions'][track_id][-1]
                    if 'frame_time' in last_desc and frame_time - last_desc['frame_time'] < 2:
                        should_generate = False
                if object_img is not None and should_generate:
                    description = generate_description(object_img, track_id)
                    description['frame_time'] = frame_time
                    description['frame_number'] = frame_number
                    if track_id not in video_data['object_descriptions']:
                        video_data['object_descriptions'][track_id] = []
                    video_data['object_descriptions'][track_id].append(description)
                    video_data['activity_timeline'].append({
                        'time': frame_time,
                        'event': 'description',
                        'frame': frame_number,
                        'description': description['english']
                    })

        annotated_frame = frame.copy()
        for det in detections:
            box = det['box']
            box_color = (0, 255, 0) if track_id is not None and det['id'] == track_id else (255, 0, 0)
            if track_id is not None and det['id'] == track_id and det.get('occluded', False):
                box_color = (0, 165, 255)
            cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), box_color, 2)
            label = f"{det['class']} {det['id']}"
            if det.get('occluded', False):
                label += " (occluded)"
            rgb_color = (box_color[2], box_color[1], box_color[0])
            annotated_frame = add_text_to_frame(annotated_frame, label, (box[0], box[1]-20), font, color=rgb_color)
            if 'conf' in det:
                conf_label = f"Conf: {det['conf']:.2f}"
                annotated_frame = add_text_to_frame(annotated_frame, conf_label, (box[0], box[3]+20), font, color=rgb_color)

        if track_id is not None:
            status_overlay = annotated_frame.copy()
            cv2.rectangle(status_overlay, (10, 10), (400, 60), (0, 0, 0), -1)
            cv2.addWeighted(status_overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
            status_txt = f"Tracking ID: {track_id}"
            if video_data['continuous_tracking']:
                status_txt += f" (Continuous Mode) - Progress: {(frame_number / video_data['total_frames'] * 100):.1f}%"
            annotated_frame = add_text_to_frame(annotated_frame, status_txt, (20, 30), font, color=(255, 255, 255))

        if track_id is not None and track_id in video_data['object_descriptions'] and video_data['object_descriptions'][track_id]:
            latest_desc = video_data['object_descriptions'][track_id][-1]
            arabic_text = latest_desc['arabic']
            english_text = latest_desc['english']
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (10, frame.shape[0]-120), (frame.shape[1]-10, frame.shape[0]-10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
            pil_img = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            wrapped_text = pil_text_wrap(english_text, font, frame.shape[1] - 40)
            y_pos = frame.shape[0] - 100
            for line in wrapped_text:
                draw.text((20, y_pos), line, font=font, fill=(255, 255, 255))
                y_pos += 25
            wrapped_ar_text = pil_text_wrap(arabic_text, font, frame.shape[1] - 40)
            y_pos = frame.shape[0] - 100 - (len(wrapped_ar_text) * 25)
            for line in wrapped_ar_text:
                reshaped_text = arabic_reshaper.reshape(line)
                bidi_text = get_display(reshaped_text)
                draw.text((20, y_pos), bidi_text, font=font, fill=(255, 255, 255))
                y_pos += 25
            annotated_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        annotated_filename = f"annotated_{int(time.time())}.jpg"
        annotated_path = os.path.join(app_config['ANNOTATED_FOLDER'], annotated_filename)
        cv2.imwrite(annotated_path, annotated_frame)
        cleanup_annotated_folder(app_config['ANNOTATED_FOLDER'])
        cap.release()

        if continuous_tracking and track_id is not None:
            if frame_number < video_data['total_frames'] - 1:
                threading.Thread(
                    target=process_next_frame,
                    args=(video_path, frame_time + (1/fps), track_id, app_config),
                    daemon=True
                ).start()
            else:
                video_data['continuous_tracking'] = False
                video_data['track_end_time'] = frame_time
                generate_tracking_summary(track_id)

        return {
            'time': frame_time,
            'annotated_image': annotated_filename,
            'detections': detections,
            'frame_width': frame.shape[1],
            'frame_height': frame.shape[0],
            'models_ready': True,
            'descriptions': video_data['object_descriptions'].get(track_id, []) if track_id else [],
            'tracking_active': video_data['continuous_tracking'],
            'progress': (frame_number / video_data['total_frames']) * 100 if video_data['total_frames'] > 0 else 0,
            'tracking_summary': video_data['tracking_summary'] if track_id else []
        }
    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()
        return {
            'time': frame_time,
            'annotated_image': None,
            'detections': [],
            'frame_width': 0,
            'frame_height': 0,
            'models_ready': models['models_loaded'],
            'error': str(e)
        }

def process_next_frame(video_path, frame_time, track_id, app_config):
    try:
        process_frame(video_path, frame_time, track_id, True, app_config)
    except Exception as e:
        print(f"Error in continuous tracking: {e}")
        video_data['continuous_tracking'] = False

def generate_tracking_summary(track_id):
    if track_id not in video_data['object_descriptions'] or not video_data['object_descriptions'][track_id]:
        video_data['tracking_summary'] = [{
            'timestamp': time.time(),
            'text': "No descriptions available for this object"
        }]
        return
    descriptions = video_data['object_descriptions'][track_id]
    timeline = video_data['activity_timeline']
    descriptions.sort(key=lambda x: x.get('frame_time', 0))
    summary_text = f"Summary for Object ID {track_id} ({video_data['all_detected_ids'][track_id]['class']}):\n\n"
    start_time = video_data['track_start_time']
    end_time = video_data['track_end_time']
    if start_time is not None and end_time is not None:
        duration = end_time - start_time
        summary_text += f"Tracked from {start_time:.2f}s to {end_time:.2f}s (Duration: {duration:.2f}s)\n\n"
    occlusion_events = [e for e in timeline if e['event'] == 'occlusion']
    if occlusion_events:
        summary_text += f"Object was occluded {len(occlusion_events)} times.\n\n"
    summary_text += "Activity Timeline:\n"
    segment_size = 5
    segments = {}
    for desc in descriptions:
        frame_time = desc.get('frame_time', 0)
        segment_key = int(frame_time // segment_size)
        if segment_key not in segments:
            segments[segment_key] = []
        segments[segment_key].append(desc)
    for segment_key in sorted(segments.keys()):
        start_time = segment_key * segment_size
        end_time = start_time + segment_size
        segment_descriptions = segments[segment_key]
        best_desc = max(segment_descriptions, key=lambda x: len(x['english']))
        summary_text += f"From {start_time:.1f}s to {end_time:.1f}s: {best_desc['english']}\n"
    summary_text += "\nComprehensive Summary:\n"
    all_descriptions = " ".join([d['english'] for d in descriptions])
    if len(all_descriptions) > 500:
        all_descriptions = all_descriptions[:497] + "..."
    summary_text += all_descriptions
    video_data['tracking_summary'] = [{
        'timestamp': time.time(),
        'text': summary_text
    }]
    video_data['full_description'] = summary_text

def cleanup_annotated_folder(annotated_folder, max_files=100):
    files = sorted(glob.glob(os.path.join(annotated_folder, 'annotated_*.jpg')),
                   key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    if len(files) > max_files:
        for old_file in files[:-max_files]:
            os.remove(old_file)

def get_video_data():
    return video_data