# app_init.py
import os
from flask import Flask
from flask_cors import CORS
from flask.cli import load_dotenv

load_dotenv()

def create_app():
    app = Flask(__name__)
    CORS(app)

    UPLOAD_FOLDER = 'uploads'
    DB_PATH = 'chroma_db'

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['DB_PATH'] = DB_PATH

    # Import & register blueprints
    from routes.upload_route import upload_bp
    app.register_blueprint(upload_bp)
    
     # Import & register query blueprint
    from routes.query_route import query_bp
    app.register_blueprint(query_bp)
    

    return app
