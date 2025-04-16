# chat/routes.py
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
import uuid
import os
from werkzeug.utils import secure_filename
import base64
from datetime import datetime

from app import db
from authentication.models import User
from chat.models import Conversation, Message, Product, Review



# Create blueprint
chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/conversations', methods=['GET'])
@jwt_required()
def get_conversations():
    """Get all conversations for the current user."""
    current_user_id = get_jwt_identity()
    user = User.query.filter_by(public_id=current_user_id).first()
    
    if not user:
        return jsonify({'error': 'User not found!'}), 404
    
    # Get all conversations for the user
    conversations = Conversation.query.filter_by(user_id=user.id).order_by(Conversation.updated_at.desc()).all()
    
    # Format the conversations
    result = []
    for conversation in conversations:
        # Get the title or first message content if title is not set
        title = conversation.title
        if not title and conversation.messages:
            first_message = sorted(conversation.messages, key=lambda m: m.timestamp)[0]
            title = first_message.content[:50] + "..." if len(first_message.content) > 50 else first_message.content
        
        result.append({
            'id': conversation.public_id,
            'title': title,
            'created_at': conversation.created_at.isoformat(),
            'updated_at': conversation.updated_at.isoformat(),
            'message_count': len(conversation.messages)
        })
    
    return jsonify(result), 200

@chat_bp.route('/conversations', methods=['POST'])
@jwt_required()
def create_conversation():
    """Create a new conversation."""
    current_user_id = get_jwt_identity()
    user = User.query.filter_by(public_id=current_user_id).first()
    
    if not user:
        return jsonify({'error': 'User not found!'}), 404
    
    data = request.json or {}
    
    # Create a new conversation
    new_conversation = Conversation(
        user_id=user.id,
        title=data.get('title'),
        metadata={
            'state': 'initial',
            'product_context': {},
            'preferences': {}
        }
    )
    
    db.session.add(new_conversation)
    db.session.commit()
    
    return jsonify({
        'id': new_conversation.public_id,
        'title': new_conversation.title,
        'created_at': new_conversation.created_at.isoformat(),
        'updated_at': new_conversation.updated_at.isoformat()
    }), 201

@chat_bp.route('/conversations/<conversation_id>', methods=['GET'])
@jwt_required()
def get_conversation(conversation_id):
    """Get a specific conversation with its messages."""
    current_user_id = get_jwt_identity()
    user = User.query.filter_by(public_id=current_user_id).first()
    
    if not user:
        return jsonify({'error': 'User not found!'}), 404
    
    # Get the conversation
    conversation = Conversation.query.filter_by(public_id=conversation_id, user_id=user.id).first()
    
    if not conversation:
        return jsonify({'error': 'Conversation not found!'}), 404
    
    # Get all messages for the conversation
    messages = Message.query.filter_by(conversation_id=conversation.id).order_by(Message.timestamp).all()
    
    # Format the messages
    formatted_messages = []
    for message in messages:
        formatted_messages.append({
            'id': message.id,
            'role': message.role,
            'content': message.content,
            'timestamp': message.timestamp.isoformat(),
            'metadata': message.metadata
        })
    
    return jsonify({
        'id': conversation.public_id,
        'title': conversation.title,
        'created_at': conversation.created_at.isoformat(),
        'updated_at': conversation.updated_at.isoformat(),
        'metadata': conversation.metadata,
        'messages': formatted_messages
    }), 200

@chat_bp.route('/conversations/<conversation_id>', methods=['PUT'])
@jwt_required()
def update_conversation(conversation_id):
    """Update a conversation's title or metadata."""
    current_user_id = get_jwt_identity()
    user = User.query.filter_by(public_id=current_user_id).first()
    
    if not user:
        return jsonify({'error': 'User not found!'}), 404
    
    # Get the conversation
    conversation = Conversation.query.filter_by(public_id=conversation_id, user_id=user.id).first()
    
    if not conversation:
        return jsonify({'error': 'Conversation not found!'}), 404
    
    data = request.json
    
    # Update conversation fields
    if 'title' in data:
        conversation.title = data['title']
    
    if 'metadata' in data:
        # Update specific metadata fields instead of replacing the entire object
        if not conversation.metadata:
            conversation.metadata = {}
        
        for key, value in data['metadata'].items():
            conversation.metadata[key] = value
    
    # Update the record
    conversation.updated_at = datetime.utcnow()
    db.session.commit()
    
    return jsonify({
        'id': conversation.public_id,
        'title': conversation.title,
        'created_at': conversation.created_at.isoformat(),
        'updated_at': conversation.updated_at.isoformat(),
        'metadata': conversation.metadata
    }), 200

@chat_bp.route('/conversations/<conversation_id>', methods=['DELETE'])
@jwt_required()
def delete_conversation(conversation_id):
    """Delete a conversation and all its messages."""
    current_user_id = get_jwt_identity()
    user = User.query.filter_by(public_id=current_user_id).first()
    
    if not user:
        return jsonify({'error': 'User not found!'}), 404
    
    # Get the conversation
    conversation = Conversation.query.filter_by(public_id=conversation_id, user_id=user.id).first()
    
    if not conversation:
        return jsonify({'error': 'Conversation not found!'}), 404
    
    # Delete the conversation (messages will be deleted due to cascade)
    db.session.delete(conversation)
    db.session.commit()
    
    return jsonify({'message': 'Conversation deleted successfully!'}), 200

@chat_bp.route('/conversations/<conversation_id>/messages', methods=['POST'])
@jwt_required()
def send_message(conversation_id):
    """Send a message in a conversation and get the chatbot's response."""
    current_user_id = get_jwt_identity()
    user = User.query.filter_by(public_id=current_user_id).first()
    
    if not user:
        return jsonify({'error': 'User not found!'}), 404
    
    # Get the conversation
    conversation = Conversation.query.filter_by(public_id=conversation_id, user_id=user.id).first()
    
    if not conversation:
        return jsonify({'error': 'Conversation not found!'}), 404
    
    data = request.json
    
    # Check if message content is provided
    if not data or not data.get('content'):
        return jsonify({'error': 'Message content is required!'}), 400
    
    # Create a new user message
    user_message = Message(
        conversation_id=conversation.id,
        role='user',
        content=data['content'],
        message_data=data.get('metadata', {})
    )
    
    db.session.add(user_message)
    
    # Update conversation timestamp
    conversation.updated_at = datetime.utcnow()
    db.session.commit()
    
    # Process image if provided
    image_description = None
    product_from_image = None
    if 'image' in data and data['image']:
        try:
            vision_analyzer = current_app.vision_analyzer
            result = vision_analyzer.process_image(data['image'])
            image_description = result['description']
            product_info = result.get('product_info', {})
            product_from_image = product_info.get('possible_product_name')
        except Exception as e:
            current_app.logger.error(f"Error processing image: {str(e)}")
    
    # Extract product from message text
    conversation_manager = current_app.conversation_manager
    product_from_text = conversation_manager.extract_product_from_message(
        data['content'],
        image_description
    )
    
    # Determine which product to review
    product_to_review = product_from_text or product_from_image
    
    # Generate appropriate response based on context
    if product_to_review:
        # Generate a review for the identified product
        prompt = f"Write a detailed review for {product_to_review}."
        review_generator = current_app.review_generator
        assistant_response = review_generator.generate_review(prompt)[0]
        
        # Clean up response
        import re
        assistant_response = re.sub(r'<[^>]*>.*?</[^>]*>', '', assistant_response)
        assistant_response = re.sub(r'<[^>]*>', '', assistant_response)
        assistant_response = re.sub(r'^[^a-zA-Z0-9]+', '', assistant_response)
    else:
        # Handle general conversation
        assistant_response = "I'm a product review chatbot. Please ask me about a specific product, and I'll generate a review for it!"
    
    # Create an assistant message with the response
    assistant_message = Message(
        conversation_id=conversation.id,
        role='assistant',
        content=assistant_response,
        message_data={
            'image_description': image_description,
            'product_reviewed': product_to_review,
            'model_version': '1.0'
        }
    )
    
    db.session.add(assistant_message)
    db.session.commit()
    
    return jsonify({
        'id': assistant_message.id,
        'role': assistant_message.role,
        'content': assistant_message.content,
        'timestamp': assistant_message.timestamp.isoformat(),
        'metadata': assistant_message.message_data
    }), 201

@chat_bp.route('/upload-image', methods=['POST'])
@jwt_required()
def upload_image():
    """Upload an image for product analysis."""
    # Check if the request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request!'}), 400
    
    file = request.files['image']
    
    # Check if a file was selected
    if file.filename == '':
        return jsonify({'error': 'No image selected!'}), 400
    
    # Check if the file is allowed
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
        return jsonify({'error': 'File type not allowed!'}), 400
    
    # Generate a secure filename and save the file
    filename = secure_filename(str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower())
    upload_folder = current_app.config['UPLOAD_FOLDER']
    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)
    
    # TODO: In a real implementation, process the image here
    # For now, we'll use a placeholder for the image description
    image_description = "Placeholder for image description"
    
    return jsonify({
        'filename': filename,
        'file_path': file_path,
        'description': image_description
    }), 201

@chat_bp.route('/reviews', methods=['GET'])
@jwt_required()
def get_reviews():
    """Get all reviews created by the current user."""
    current_user_id = get_jwt_identity()
    user = User.query.filter_by(public_id=current_user_id).first()
    
    if not user:
        return jsonify({'error': 'User not found!'}), 404
    
    # Get all reviews for the user
    reviews = Review.query.filter_by(user_id=user.id).order_by(Review.created_at.desc()).all()
    
    # Format the reviews
    result = []
    for review in reviews:
        product = Product.query.get(review.product_id)
        
        result.append({
            'id': review.id,
            'product': {
                'id': product.id,
                'name': product.name,
                'category': product.category
            },
            'content': review.content,
            'rating': review.rating,
            'created_at': review.created_at.isoformat(),
            'metadata': review.metadata
        })
    
    return jsonify(result), 200

@chat_bp.route('/reviews', methods=['POST'])
@jwt_required()
def create_review():
    """Create a new product review."""
    current_user_id = get_jwt_identity()
    user = User.query.filter_by(public_id=current_user_id).first()
    
    if not user:
        return jsonify({'error': 'User not found!'}), 404
    
    data = request.json
    
    # Check if required fields are present
    if not data or not data.get('product_name') or not data.get('content'):
        return jsonify({'error': 'Product name and review content are required!'}), 400
    
    # Check if the product exists, or create a new one
    product = Product.query.filter_by(name=data['product_name']).first()
    
    if not product:
        # Create a new product
        product = Product(
            name=data['product_name'],
            category=data.get('category'),
            description=data.get('description')
        )
        db.session.add(product)
        db.session.commit()
    
    # Create a new review
    new_review = Review(
        product_id=product.id,
        user_id=user.id,
        conversation_id=data.get('conversation_id'),
        content=data['content'],
        rating=data.get('rating'),
        metadata=data.get('metadata', {})
    )
    
    db.session.add(new_review)
    db.session.commit()
    
    return jsonify({
        'id': new_review.id,
        'product': {
            'id': product.id,
            'name': product.name,
            'category': product.category
        },
        'content': new_review.content,
        'rating': new_review.rating,
        'created_at': new_review.created_at.isoformat(),
        'metadata': new_review.metadata
    }), 201