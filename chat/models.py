from datetime import datetime
from app import db
import uuid

class Conversation(db.Model):
    """Model for user conversations."""
    __tablename__ = 'conversations'
    
    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(50), unique=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship with messages
    messages = db.relationship('Message', backref='conversation', lazy=True, cascade='all, delete-orphan')
    
    # Conversation metadata
    context_data = db.Column(db.JSON, nullable=True)
    
    def __repr__(self):
        return f'<Conversation {self.public_id}>'

class Message(db.Model):
    """Model for chat messages."""
    __tablename__ = 'messages'
    
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversations.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # e.g., 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Message metadata (e.g., for storing referenced products, sentiment, etc.)
    message_data = db.Column(db.JSON, nullable=True)
    
    def __repr__(self):
        return f'<Message {self.id} ({self.role})>'

class Product(db.Model):
    """Model for products discussed in conversations."""
    __tablename__ = 'products'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(50), nullable=True)
    description = db.Column(db.Text, nullable=True)
    image_url = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship with reviews
    reviews = db.relationship('Review', backref='product', lazy=True)
    
    def __repr__(self):
        return f'<Product {self.name}>'

class Review(db.Model):
    """Model for product reviews."""
    __tablename__ = 'reviews'
    
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversations.id'), nullable=True)
    content = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime)
    
    # Review metadata
    review_data = db.Column(db.JSON, nullable=True)
    
    def __repr__(self):
        return f'<Review {self.id} for Product {self.product_id}>'

