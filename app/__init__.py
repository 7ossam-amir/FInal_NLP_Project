from flask import Flask
import os

def create_app():
    app = Flask(__name__, static_folder='static', template_folder='templates')
    
    # Configuration
    app.config.from_object('app.config.Config')
    
    # Create directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['ANNOTATED_FOLDER'], exist_ok=True)
    
    # Register routes
    from .routes import init_routes
    init_routes(app)
    
    return app