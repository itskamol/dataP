"""
Authentication and authorization middleware.
Implements requirements 6.4, 7.2: API authentication and authorization with JWT tokens.
"""

import os
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from functools import wraps

from flask import Flask, request, jsonify, current_app
from flask_jwt_extended import JWTManager, verify_jwt_in_request, get_jwt_identity, get_jwt
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import Unauthorized, Forbidden, TooManyRequests

from src.infrastructure.logging import get_logger

logger = get_logger('auth_middleware')


class AuthenticationManager:
    """Manages JWT authentication and user validation."""
    
    def __init__(self, app: Optional[Flask] = None):
        """Initialize authentication manager."""
        self.jwt = JWTManager()
        self.users_db = {}  # In production, use proper user database
        
        if app:
            self.init_app(app)
    
    def init_app(self, app: Flask):
        """Initialize with Flask app."""
        # JWT Configuration
        app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'dev-secret-key-change-in-production')
        app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
        app.config['JWT_ALGORITHM'] = 'HS256'
        
        self.jwt.init_app(app)
        
        # JWT error handlers
        @self.jwt.expired_token_loader
        def expired_token_callback(jwt_header, jwt_payload):
            logger.warning(f"Expired token access attempt from {request.remote_addr}")
            return jsonify({
                'error': 'Token has expired',
                'details': 'Please login again to get a new token',
                'timestamp': datetime.now().isoformat()
            }), 401
        
        @self.jwt.invalid_token_loader
        def invalid_token_callback(error):
            logger.warning(f"Invalid token access attempt from {request.remote_addr}: {error}")
            return jsonify({
                'error': 'Invalid token',
                'details': 'Token is malformed or invalid',
                'timestamp': datetime.now().isoformat()
            }), 401
        
        @self.jwt.unauthorized_loader
        def missing_token_callback(error):
            logger.warning(f"Missing token access attempt from {request.remote_addr}")
            return jsonify({
                'error': 'Authorization token required',
                'details': 'Please provide a valid JWT token in the Authorization header',
                'timestamp': datetime.now().isoformat()
            }), 401
        
        @self.jwt.revoked_token_loader
        def revoked_token_callback(jwt_header, jwt_payload):
            logger.warning(f"Revoked token access attempt from {request.remote_addr}")
            return jsonify({
                'error': 'Token has been revoked',
                'details': 'This token is no longer valid',
                'timestamp': datetime.now().isoformat()
            }), 401
        
        # Initialize demo users (in production, load from database)
        self._init_demo_users()
    
    def _init_demo_users(self):
        """Initialize demo users for testing."""
        demo_users = [
            {'username': 'admin', 'password': 'admin123', 'role': 'admin'},
            {'username': 'user', 'password': 'user123', 'role': 'user'},
            {'username': 'demo', 'password': 'demo123', 'role': 'user'}
        ]
        
        for user in demo_users:
            password_hash = self._hash_password(user['password'])
            self.users_db[user['username']] = {
                'password_hash': password_hash,
                'role': user['role'],
                'created_at': datetime.now(),
                'active': True
            }
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256 (in production, use bcrypt or similar)."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def validate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Validate user credentials."""
        if username not in self.users_db:
            return None
        
        user_data = self.users_db[username]
        if not user_data.get('active', False):
            return None
        
        password_hash = self._hash_password(password)
        if password_hash != user_data['password_hash']:
            return None
        
        return {
            'username': username,
            'role': user_data['role'],
            'active': user_data['active']
        }
    
    def get_user_info(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user information."""
        if username not in self.users_db:
            return None
        
        user_data = self.users_db[username]
        return {
            'username': username,
            'role': user_data['role'],
            'active': user_data['active'],
            'created_at': user_data['created_at'].isoformat()
        }


class RateLimitManager:
    """Manages API rate limiting and request throttling."""
    
    def __init__(self, app: Optional[Flask] = None):
        """Initialize rate limit manager."""
        self.limiter = None
        
        if app:
            self.init_app(app)
    
    def init_app(self, app: Flask):
        """Initialize with Flask app."""
        # Rate limiting configuration
        app.config['RATELIMIT_STORAGE_URL'] = os.getenv('REDIS_URL', 'memory://')
        app.config['RATELIMIT_STRATEGY'] = 'fixed-window'
        app.config['RATELIMIT_HEADERS_ENABLED'] = True
        
        self.limiter = Limiter(
            key_func=self._get_rate_limit_key,
            default_limits=["1000 per hour", "100 per minute"],
            headers_enabled=True
        )
        self.limiter.init_app(app)
        
        # Rate limit error handler
        @app.errorhandler(429)
        def ratelimit_handler(e):
            logger.warning(f"Rate limit exceeded from {request.remote_addr}")
            return jsonify({
                'error': 'Rate limit exceeded',
                'details': 'Too many requests. Please try again later.',
                'retry_after': getattr(e, 'retry_after', None),
                'timestamp': datetime.now().isoformat()
            }), 429
    
    def _get_rate_limit_key(self) -> str:
        """Get rate limiting key based on user or IP."""
        try:
            # Try to get authenticated user
            verify_jwt_in_request(optional=True)
            user = get_jwt_identity()
            if user:
                return f"user:{user}"
        except:
            pass
        
        # Fall back to IP address
        return f"ip:{get_remote_address()}"
    
    def get_limits(self) -> Dict[str, str]:
        """Get current rate limits."""
        return {
            'default': "1000 per hour, 100 per minute",
            'auth': "2000 per hour, 200 per minute",
            'upload': "50 per hour, 10 per minute",
            'processing': "20 per hour, 5 per minute"
        }


def require_role(required_role: str):
    """Decorator to require specific user role."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                verify_jwt_in_request()
                current_user = get_jwt_identity()
                jwt_data = get_jwt()
                
                user_role = jwt_data.get('role', 'user')
                
                # Define role hierarchy
                role_hierarchy = {
                    'user': 0,
                    'admin': 1
                }
                
                required_level = role_hierarchy.get(required_role, 0)
                user_level = role_hierarchy.get(user_role, 0)
                
                if user_level < required_level:
                    logger.warning(f"Insufficient permissions for user {current_user}: required {required_role}, has {user_role}")
                    raise Forbidden(f"Insufficient permissions. Required role: {required_role}")
                
                return f(*args, **kwargs)
                
            except Forbidden:
                raise
            except Exception as e:
                logger.error(f"Role check error: {str(e)}")
                raise Unauthorized("Authentication required")
        
        return decorated_function
    return decorator


def log_api_access(f):
    """Decorator to log API access."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = datetime.now()
        
        try:
            # Get user info if authenticated
            user = "anonymous"
            try:
                verify_jwt_in_request(optional=True)
                user = get_jwt_identity() or "anonymous"
            except:
                pass
            
            # Log request
            logger.info(f"API access: {request.method} {request.path} by {user} from {request.remote_addr}")
            
            # Execute function
            result = f(*args, **kwargs)
            
            # Log success
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"API success: {request.method} {request.path} completed in {duration:.3f}s")
            
            return result
            
        except Exception as e:
            # Log error
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"API error: {request.method} {request.path} failed in {duration:.3f}s: {str(e)}")
            raise
    
    return decorated_function


def validate_content_type(allowed_types: List[str]):
    """Decorator to validate request content type."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if request.method in ['POST', 'PUT', 'PATCH']:
                content_type = request.content_type
                if content_type:
                    # Extract main content type (ignore charset, etc.)
                    main_type = content_type.split(';')[0].strip()
                    if main_type not in allowed_types:
                        logger.warning(f"Invalid content type {content_type} from {request.remote_addr}")
                        return jsonify({
                            'error': 'Invalid content type',
                            'details': f'Expected one of: {", ".join(allowed_types)}',
                            'timestamp': datetime.now().isoformat()
                        }), 400
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


def validate_request_size(max_size_mb: int = 100):
    """Decorator to validate request size."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if request.content_length:
                max_size_bytes = max_size_mb * 1024 * 1024
                if request.content_length > max_size_bytes:
                    logger.warning(f"Request too large ({request.content_length} bytes) from {request.remote_addr}")
                    return jsonify({
                        'error': 'Request too large',
                        'details': f'Maximum request size is {max_size_mb}MB',
                        'timestamp': datetime.now().isoformat()
                    }), 413
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


# Global instances
auth_manager = AuthenticationManager()
rate_limit_manager = RateLimitManager()