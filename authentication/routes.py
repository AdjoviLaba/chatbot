from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import uuid

from app import db
from authentication.models import User

# Create blueprint
auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['POST'])
def register():
    """Register a new user."""
    data = request.json
    
    # Check if required fields are present
    required_fields = ['username', 'email', 'password']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    # Check if username already exists
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already exists!'}), 409
    
    # Check if email already exists
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already registered!'}), 409
    
    # Create a new user
    hashed_password = generate_password_hash(data['password'], method='pbkdf2:sha256')
    new_user = User(
        username=data['username'],
        email=data['email'],
        password=hashed_password
    )
    
    # Add user to database
    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({
        'message': 'User registered successfully!',
        'user_id': new_user.public_id
    }), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    """Log in a user and return an access token."""
    data = request.json
    
    # Check if required fields are present
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'error': 'Missing username or password!'}), 400
    
    # Find user by username
    user = User.query.filter_by(username=data['username']).first()
    
    # Check if user exists and password is correct
    if not user or not check_password_hash(user.password, data['password']):
        return jsonify({'error': 'Invalid username or password!'}), 401
    
    # Update last login time
    user.last_login = datetime.utcnow()
    db.session.commit()
    
    # Create access token
    access_token = create_access_token(identity=user.public_id)
    
    return jsonify({
        'message': 'Login successful!',
        'access_token': access_token,
        'user_id': user.public_id,
        'username': user.username
    }), 200

@auth_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """Get the current user's profile."""
    current_user_id = get_jwt_identity()
    user = User.query.filter_by(public_id=current_user_id).first()
    
    if not user:
        return jsonify({'error': 'User not found!'}), 404
    
    return jsonify({
        'public_id': user.public_id,
        'username': user.username,
        'email': user.email,
        'created_at': user.created_at.isoformat(),
        'last_login': user.last_login.isoformat() if user.last_login else None
    }), 200

@auth_bp.route('/change-password', methods=['POST'])
@jwt_required()
def change_password():
    """Change the current user's password."""
    current_user_id = get_jwt_identity()
    user = User.query.filter_by(public_id=current_user_id).first()
    
    if not user:
        return jsonify({'error': 'User not found!'}), 404
    
    data = request.json
    
    # Check if required fields are present
    if not data or not data.get('current_password') or not data.get('new_password'):
        return jsonify({'error': 'Missing current password or new password!'}), 400
    
    # Verify current password
    if not check_password_hash(user.password, data['current_password']):
        return jsonify({'error': 'Current password is incorrect!'}), 401
    
    # Update password
    user.password = generate_password_hash(data['new_password'], method='pbkdf2:sha256')
    db.session.commit()
    
    return jsonify({'message': 'Password changed successfully!'}), 200

@auth_bp.route('/reset-password-request', methods=['POST'])
def reset_password_request():
    """Request a password reset."""
    data = request.json
    
    # Check if email is provided
    if not data or not data.get('email'):
        return jsonify({'error': 'Email is required!'}), 400
    
    # Find user by email
    user = User.query.filter_by(email=data['email']).first()
    
    if not user:
        # Don't reveal that the user doesn't exist
        return jsonify({'message': 'Email not found.'}), 200
    

    reset_token = create_access_token(
        identity=user.public_id,
        additional_claims={"reset_password": True}
    )
    
    return jsonify({
        'message': 'Password reset requested. Check your email for instructions.',
        'reset_token': reset_token  # Remove this in production
    }), 200

@auth_bp.route('/reset-password', methods=['POST'])
def reset_password():
    """Reset a user's password using a reset token."""
    data = request.json
    
    # Check if required fields are present
    if not data or not data.get('reset_token') or not data.get('new_password'):
        return jsonify({'error': 'Reset token and new password are required!'}), 400
    
    try:
        # Verify the reset token
        from flask_jwt_extended import decode_token
        token_data = decode_token(data['reset_token'])
        
        # Check if this is a password reset token
        if not token_data.get('reset_password', False):
            return jsonify({'error': 'Invalid reset token!'}), 401
        
        # Find the user
        user = User.query.filter_by(public_id=token_data['sub']).first()
        
        if not user:
            return jsonify({'error': 'User not found!'}), 404
        
        # Update password
        user.password = generate_password_hash(data['new_password'], method='pbkdf2:sha256')
        db.session.commit()
        
        return jsonify({'message': 'Password reset successful!'}), 200
        
    except Exception as e:
        current_app.logger.error(f"Password reset error: {str(e)}")
        return jsonify({'error': 'Invalid or expired reset token!'}), 401