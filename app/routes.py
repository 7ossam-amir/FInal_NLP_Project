from flask import render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from .video_processing import get_video_data, get_video_info, process_frame, allowed_file, cleanup_annotated_folder, generate_tracking_summary
import os
import glob
import time  # Added missing import
import app.gemini_processing as gp

def init_routes(app):
    video_data = get_video_data()

    @app.route('/')
    def index():
        return redirect(url_for('upload_page'))

    @app.route('/upload', methods=['GET', 'POST'])
    def upload_page():
        if request.method == 'POST':
            return redirect(url_for('upload'))
        video_info = get_video_info(video_data['video_path'])
        return render_template('upload.html', video_info=video_info)

    @app.route('/upload_file', methods=['POST'])
    def upload():
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            video_data['filename'] = filename
            video_data['video_path'] = filepath
            video_data['selected_id'] = None
            video_data['track_history'] = []
            video_data['all_detected_ids'] = {}
            video_data['object_descriptions'] = {}
            video_data['object_positions'] = {}
            video_data['occlusion_count'] = {}
            video_data['continuous_tracking'] = False
            video_data['track_end_time'] = None
            video_data['track_start_time'] = None
            video_data['full_description'] = ''
            video_data['tracking_active'] = False
            video_data['tracking_summary'] = []
            video_data['activity_timeline'] = []
            return redirect(url_for('play'))
        return redirect(request.url)

    @app.route('/play')
    def play():
        if not video_data['video_path'] or not os.path.exists(video_data['video_path']):
            return redirect(url_for('upload_page'))
        video_info = get_video_info(video_data['video_path'])
        return render_template('play.html', video_info=video_info)

    @app.route('/uploads/<filename>')
    def uploaded_file(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    @app.route('/annotated/<filename>')
    def annotated_file(filename):
        return send_from_directory(app.config['ANNOTATED_FOLDER'], filename)

    @app.route('/process_frame', methods=['POST'])
    def process_frame_endpoint():
        data = request.json
        if not video_data['video_path']:
            return jsonify({'error': 'No video loaded'})
        frame_time = data.get('time', 0)
        result = process_frame(video_data['video_path'], frame_time, app_config=app.config)
        return jsonify(result)

    @app.route('/start_tracking', methods=['POST'])
    def start_tracking():
        data = request.json
        if not video_data['video_path']:
            return jsonify({'error': 'No video loaded'})
        track_id = data.get('track_id')
        if track_id is None:
            return jsonify({'error': 'No object ID specified for tracking'})
        frame_time = data.get('time', 0)
        video_data['selected_id'] = track_id
        video_data['track_start_time'] = frame_time
        video_data['continuous_tracking'] = True
        video_data['tracking_active'] = True
        video_data['track_history'] = []
        video_data['occlusion_count'] = {}
        video_data['object_descriptions'] = {track_id: []} if track_id not in video_data['object_descriptions'] else video_data['object_descriptions']
        video_data['activity_timeline'] = []
        result = process_frame(video_data['video_path'], frame_time, track_id, True, app_config=app.config)
        return jsonify(result)

    @app.route('/stop_tracking', methods=['POST'])
    def stop_tracking():
        video_data['continuous_tracking'] = False
        video_data['tracking_active'] = False
        violence_report = None

        if video_data['selected_id'] is not None:
            # Generate tracking summary
            video_data['tracking_summary'] = generate_tracking_summary(video_data['selected_id'])

            # Debug: Print tracking summary
            print("Tracking summary for ID", video_data['selected_id'], ":", video_data['tracking_summary'])

            # Generate violence report using the tracking summary
            if video_data['tracking_summary']:
                violence_report = gp.violence_report_generator(video_data['tracking_summary'][0]['text'] if video_data['tracking_summary'] else '')
                violence_report_filename = f"violence_report_{video_data['filename']}_{video_data['selected_id']}.txt"
                violence_report_filepath = os.path.join(app.config['UPLOAD_FOLDER'], violence_report_filename)
                with open(violence_report_filepath, 'w') as f:
                    f.write(violence_report)

        return jsonify({
            'success': True,
            'summary': video_data['tracking_summary'],
            'violence_report': violence_report  # Return the report content
        })

    @app.route('/tracking_status', methods=['GET'])
    def tracking_status():
        if not video_data['tracking_active'] and not video_data['continuous_tracking']:
            return jsonify({
                'tracking_active': False,
                'selected_id': video_data['selected_id'],
                'progress': 0,
                'tracking_summary': video_data['tracking_summary']
            })
        track_id = video_data['selected_id']
        result = {
            'tracking_active': video_data['continuous_tracking'],
            'selected_id': track_id,
            'progress': (video_data['current_frame'] / video_data['total_frames'] * 100) if video_data['total_frames'] > 0 else 0,
            'descriptions': video_data['object_descriptions'].get(track_id, []),
            'annotated_image': None,
            'tracking_summary': video_data['tracking_summary']
        }
        annotated_files = sorted(glob.glob(os.path.join(app.config['ANNOTATED_FOLDER'], 'annotated_*.jpg')),
                               key=lambda x: os.path.getmtime(x), reverse=True)
        if annotated_files:
            result['annotated_image'] = os.path.basename(annotated_files[0])
        return jsonify(result)

    @app.route('/generate_analysis', methods=['POST'])
    def generate_analysis():
        data = request.get_json()
        summary = data.get('summary', '')
        
        if not summary:
            return jsonify({'error': 'No summary provided'}), 400
        
        try:
            # Use the threat analyzer to generate a report
            report = gp.violence_report_generator(summary)
            
            # Save the report to a file for reference
            analysis_filename = f"analysis_report_{time.time()}.txt"
            analysis_filepath = os.path.join(app.config['UPLOAD_FOLDER'], analysis_filename)
            with open(analysis_filepath, 'w') as f:
                f.write(report)
            
            return jsonify({
                'success': True,
                'report': report
            }), 200
        except Exception as e:
            print(f"Error generating analysis: {str(e)}")
            return jsonify({'error': f"Failed to generate analysis: {str(e)}"}), 500