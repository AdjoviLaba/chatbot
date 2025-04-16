import os
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
from config import get_config

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()
jwt = JWTManager()

def register_ml_components(app):
    """Register machine learning components with the app."""
    from machine_learning.conversation_saver import ConversationManager
    from machine_learning.generative_text import ReviewGenerator
    from machine_learning.vision_model import ProductVisionAnalyzer
    
    app.conversation_manager = ConversationManager()
    app.review_generator = ReviewGenerator(model_path="models/review_gpt_model")
    app.vision_analyzer = ProductVisionAnalyzer()
    
    app.logger.info("ML components initialized successfully")

def create_app(config_object=None):

    app = Flask(__name__)
    
    if config_object is None:
        config = get_config()
        app.config.from_object(config)
    else:
        app.config.from_object(config_object)
    
    CORS(app, supports_credentials=True)

    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)
    
 
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    

    from authentication.routes import auth_bp
    from chat.routes import chat_bp
    from test_routes import test_bp
    
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(chat_bp, url_prefix='/api/chat')
    app.register_blueprint(test_bp, url_prefix='/api/test')
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    # Register ML components
    register_ml_components(app)
    
    # Add home route
    @app.route('/')
    def home():
        """Render the homepage with the chatbot interface."""
        return render_template('index.html')
    
    return app

if __name__ == '__main__':
    app = create_app()
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=app.config['DEBUG'])