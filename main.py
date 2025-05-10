import os
from pyngrok import ngrok, conf
from app import create_app

# Replace with your valid ngrok auth token
NGROK_AUTH_TOKEN = '2vmgsDLjGCur8P5JF1LXquHmNL9_x31q3baL72mbp2ATw3Zg'

def start_ngrok():
    """Start ngrok for Colab."""
    try:
        conf.get_default().auth_token = NGROK_AUTH_TOKEN
        public_url = ngrok.connect(5000).public_url
        print(f" * Running on {public_url}")
        return public_url
    except Exception as e:
        print(f" * Failed to start ngrok: {e}")
        return None

if __name__ == '__main__':
    # Create Flask app
    app = create_app()

    # Start ngrok if running in Colab
    if 'COLAB_GPU' in os.environ:
        start_ngrok()

    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)