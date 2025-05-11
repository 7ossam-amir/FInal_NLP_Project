# Advanced Video Tracker with Threat Analysis

## Project Overview

The **Advanced Video Tracker** is a Flask-based web application designed to track objects in videos, generate summaries of tracking activities, and perform threat analysis using Natural Language Processing (NLP) techniques. The project leverages computer vision for object detection and tracking, NLP for generating summaries, and a Large Language Model (LLM) for refining summaries before conducting threat analysis. Key features include:

- **Object Tracking**: Upload a video, analyze frames, and track specific objects with bounding box annotations.
- **Tracking Summary**: Generate summaries of tracked object activities, such as movement and interactions.
- **LLM Refinement**: Refine tracking summaries using an LLM to improve clarity and coherence.
- **Threat Analysis**: Analyze refined summaries to assess potential threats (e.g., violence detection) with a detailed report.
- **Interactive UI**: A user-friendly interface to upload videos, view tracking results, and generate analysis reports.

The project integrates computer vision libraries (e.g., OpenCV), NLP models (e.g., BERT for multilabel classification), and Flask for the web framework. It also includes a pre-trained BERT model for threat analysis, which is downloaded during setup.

## Project Structure

- **app/**: Contains the Flask application code.
  - `routes.py`: Defines Flask routes for video upload, tracking, and analysis.
  - `video_processing.py`: Handles video frame processing and object tracking.
  - `gemini_processing.py`: Manages NLP tasks like violence report generation.
  - `threat_analyzer.py`: Performs threat analysis on tracking summaries.
  - `templates/`: HTML templates (`upload.html`, `play.html`) for the web interface.
- **static/**: Stores static assets like CSS, JavaScript, and images.
- **uploads/**: Directory for uploaded videos.
- **annotated/**: Directory for annotated frames.
- `requirements.txt`: Lists Python dependencies.
- `fine_tuned_bert_multilabel.pt`: Pre-trained BERT model for threat analysis.
- `README.md`: Project documentation.

## Prerequisites

- Python 3.8 or higher
- Git
- A compatible environment (local machine or Jupyter Notebook/Google Colab)
- Internet access (for downloading the pre-trained model and dependencies)

## Setup and Installation

### Running Locally

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/7ossam-amir/Final_NLP_Project.git
   mv Final_NLP_Project NLP_FINAL_PR
   cd NLP_FINAL_PR
   ```

2. **Set Up a Virtual Environment** (Optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   pip install gdown
   ```

4. **Download the Pre-trained BERT Model**:

   ```bash
   python -c "import gdown; gdown.download('https://drive.google.com/uc?export=download&id=17fKW27lHLiPJK9KT7mN9PUQQIpUD93LN', 'fine_tuned_bert_multilabel.pt', quiet=False)"
   ```

5. **Run the Flask Application**:

   ```bash
   export FLASK_APP=app  # On Windows: set FLASK_APP=app
   export FLASK_ENV=development  # On Windows: set FLASK_ENV=development
   flask run
   ```

   - The app will be available at `http://127.0.0.1:5000`.

6. **Access the Application**:

   - Open your browser and navigate to `http://127.0.0.1:5000`.
   - Upload a video, track objects, and generate threat analysis reports.

### Running in a Jupyter Notebook (e.g., Google Colab)

1. **Clone the Repository**:

   ```python
   !git clone https://github.com/7ossam-amir/Final_NLP_Project.git
   !mv Final_NLP_Project NLP_FINAL_PR
   %cd NLP_FINAL_PR
   ```

2. **Install Dependencies**:

   ```python
   !pip install -r requirements.txt
   !pip install gdown
   ```

3. **Download the Pre-trained BERT Model**:

   ```python
   import gdown
   file_id = '17fKW27lHLiPJK9KT7mN9PUQQIpUD93LN'
   url = f'https://drive.google.com/uc?export=download&id={file_id}'
   output_path = 'fine_tuned_bert_multilabel.pt'
   gdown.download(url, output_path, quiet=False)
   ```

4. **Run the Flask Application**:

   ```python
   import gdown
   file_id = '17fKW27lHLiPJK9KT7mN9PUQQIpUD93LN'
   url = f'https://drive.google.com/uc?export=download&id={file_id}'
   output_path = 'fine_tuned_bert_multilabel.pt'
   gdown.download(url, output_path, quiet=False)
   ```
   
 

5. **Running Flask in Colab (Optional)**:

   - Use `ngrok` to expose the Flask app:

     ```python
      !python /content/NLP_FINAL_PR/main.py
     ```

   - Follow the ngrok URL provided in the output to access the app.

## Usage

1. **Upload a Video**:

   - Navigate to the upload page, select a video file (e.g., MP4), and upload it.

2. **Track Objects**:

   - Play the video, analyze a frame, select an object to track, and start tracking.
   - Stop tracking to generate a summary of the object's activities.

3. **Generate Threat Analysis**:

   - View the tracking summary in the modal.
   - Click "See Report Analysis" to refine the summary using an LLM and generate a threat analysis report.
   - The report will include the original summary, refined summary, and threat analysis details (e.g., risk levels).


## Troubleshooting

- **Flask Not Running**: Ensure `FLASK_APP` is set and all dependencies are installed.
- **Model Download Fails**: Verify the Google Drive link and `gdown` installation.
- **LLM Refinement Fails**: Check API keys, network connectivity, and error logs in `routes.py`.
- **Threat Analysis Errors**: Ensure the `fine_tuned_bert_multilabel.pt` file is in the project root and compatible with your environment.

## Future Improvements

- Integrate real-time tracking with WebSocket for a smoother UI experience.
- Add support for multiple object tracking simultaneously.
- Enhance threat analysis with more advanced NLP models.
- Deploy the application on a cloud platform (e.g., Heroku, AWS).

