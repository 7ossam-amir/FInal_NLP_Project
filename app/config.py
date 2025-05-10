class Config:
    UPLOAD_FOLDER = '/content/NLP_FINAL_PR/app/uploads'
    ANNOTATED_FOLDER = 'app/annotated'
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB limit
    NGROK_AUTH_TOKEN = '2vmgsDLjGCur8P5JF1LXquHmNL9_x31q3baL72mbp2ATw3Zg'