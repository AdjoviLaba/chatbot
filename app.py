"""
Application Initialization Script
--------------------------------
Initializes the Flask application with all necessary components.
"""

import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
from config import get_config

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()
jwt = JWTManager()

def create_app(config_object=None):


    # Create Flask app
    app = Flask(__name__)
    
    # Load configuration
    if config_object is None:
        config = get_config()
        app.config.from_object(config)
    else:
        app.config.from_object(config_object)
    
    # Enable CORS
    CORS(app, supports_credentials=True)
    
    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)

    # Define a simple home route
    @app.route('/')
    def home():
        return {
            'status': 'online',
            'app': 'Product Review Chatbot API',
            'endpoints': {
                'auth': '/api/auth/*',
                'chat': '/api/chat/*'
            }
        }


    # Create upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    

    # Register blueprints
    from authentication.routes import auth_bp
    from chat.routes import chat_bp
    
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(chat_bp, url_prefix='/api/chat')
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    return app

if __name__ == '__main__':
    app = create_app()
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=app.config['DEBUG'])