"""
Flask application factory for REST API.
Implements requirements 6.1, 6.4, 7.2, 4.1: REST API with authentication, rate limiting, and documentation.
"""

import os
from datetime import timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_jwt_extended.exceptions import NoAuthorizationError
from jwt.exceptions import DecodeError


from src.web.routes.api_routes import api_bp
from src.web.middleware.auth_middleware import auth_manager, rate_limit_manager
from src.web.services.job_service import job_service
from src.infrastructure.logging import get_logger

logger = get_logger('api_app')


def create_api_app(config_name: str = 'development') -> Flask:
    """
    Create and configure Flask API application.
    
    Args:
        config_name: Configuration environment name
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Load configuration
    configure_app(app, config_name)
    
    # Logging is already set up by the logger manager
    
    # Initialize extensions
    initialize_extensions(app)
    
    # Register blueprints
    register_blueprints(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Add health check endpoint
    register_health_check(app)
    
    logger.info(f"API application created with config: {config_name}")
    return app


def configure_app(app: Flask, config_name: str):
    """Configure Flask application."""
    
    # Base configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['DEBUG'] = config_name == 'development'
    app.config['TESTING'] = config_name == 'testing'
    
    # JWT Configuration
    app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'jwt-secret-key-change-in-production')
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=int(os.getenv('JWT_EXPIRES_HOURS', '24')))
    app.config['JWT_ALGORITHM'] = 'HS256'
    app.config['JWT_BLACKLIST_ENABLED'] = True
    app.config['JWT_BLACKLIST_TOKEN_CHECKS'] = ['access']
    
    # File upload configuration
    app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_UPLOAD_SIZE', '104857600'))  # 100MB
    
    # Rate limiting configuration
    app.config['RATELIMIT_STORAGE_URL'] = os.getenv('REDIS_URL', 'memory://')
    app.config['RATELIMIT_STRATEGY'] = 'fixed-window'
    app.config['RATELIMIT_HEADERS_ENABLED'] = True
    
    # CORS configuration
    app.config['CORS_ORIGINS'] = os.getenv('CORS_ORIGINS', '*').split(',')
    app.config['CORS_METHODS'] = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
    app.config['CORS_HEADERS'] = ['Content-Type', 'Authorization']
    
    # Logging configuration
    app.config['LOG_LEVEL'] = os.getenv('LOG_LEVEL', 'INFO')
    app.config['LOG_FORMAT'] = os.getenv('LOG_FORMAT', 'json')
    
    # API Documentation
    app.config['RESTX_MASK_SWAGGER'] = False
    app.config['RESTX_VALIDATE'] = True
    app.config['RESTX_JSON'] = {'ensure_ascii': False, 'indent': 2}
    
    # Environment-specific configurations
    if config_name == 'development':
        app.config['DEBUG'] = True
        app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
        
    elif config_name == 'testing':
        app.config['TESTING'] = True
        app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(minutes=15)
        app.config['UPLOAD_FOLDER'] = '/tmp/test_uploads'
        app.config['RATELIMIT_STORAGE_URL'] = 'memory://'
        
    elif config_name == 'production':
        app.config['DEBUG'] = False
        app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
        
        # Production security settings
        app.config['SESSION_COOKIE_SECURE'] = True
        app.config['SESSION_COOKIE_HTTPONLY'] = True
        app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
        
        # Require HTTPS in production
        if not os.getenv('ALLOW_HTTP'):
            app.config['PREFERRED_URL_SCHEME'] = 'https'


def initialize_extensions(app: Flask):
    """Initialize Flask extensions."""
    
    # CORS
    CORS(app, 
         origins=app.config['CORS_ORIGINS'],
         methods=app.config['CORS_METHODS'],
         allow_headers=app.config['CORS_HEADERS'],
         supports_credentials=True)
    
    # Authentication and rate limiting
    auth_manager.init_app(app)
    rate_limit_manager.init_app(app)
    
    # Create upload directory
    upload_folder = app.config['UPLOAD_FOLDER']
    os.makedirs(upload_folder, exist_ok=True)
    
    # Start job service
    job_service.start()
    
    logger.info("Extensions initialized successfully")


def register_blueprints(app: Flask):
    """Register Flask blueprints."""
    
    # Register API blueprint
    app.register_blueprint(api_bp)
    
    logger.info("Blueprints registered successfully")


def register_error_handlers(app: Flask):
    """Register global error handlers."""
    
    @app.errorhandler(400)
    def bad_request(error):
        """Handle bad request errors."""
        return jsonify({
            'error': 'Bad Request',
            'details': str(error.description) if error.description else 'Invalid request',
            'timestamp': '2025-01-27T00:00:00Z'
        }), 400
    
    @app.errorhandler(401)
    def unauthorized(error):
        """Handle unauthorized errors."""
        return jsonify({
            'error': 'Unauthorized',
            'details': 'Authentication required',
            'timestamp': '2025-01-27T00:00:00Z'
        }), 401
    
    @app.errorhandler(NoAuthorizationError)
    def handle_no_authorization_error(error):
        """Handle missing authorization header errors."""
        logger.warning(f"Missing authorization header from {request.remote_addr}: {str(error)}")
        return jsonify({
            'error': 'Unauthorized',
            'details': 'Authorization header is required. Please provide a valid JWT token.',
            'timestamp': '2025-01-27T00:00:00Z'
        }), 401
    
    @app.errorhandler(DecodeError)
    def handle_decode_error(error):
        """Handle JWT decode errors."""
        logger.warning(f"JWT decode error from {request.remote_addr}: {str(error)}")
        return jsonify({
            'error': 'Unauthorized',
            'details': 'Invalid token format.',
            'timestamp': '2025-01-27T00:00:00Z'
        }), 401
    
    @app.errorhandler(403)
    def forbidden(error):
        """Handle forbidden errors."""
        return jsonify({
            'error': 'Forbidden',
            'details': 'Insufficient permissions',
            'timestamp': '2025-01-27T00:00:00Z'
        }), 403
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle not found errors."""
        return jsonify({
            'error': 'Not Found',
            'details': 'The requested resource was not found',
            'timestamp': '2025-01-27T00:00:00Z'
        }), 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        """Handle method not allowed errors."""
        return jsonify({
            'error': 'Method Not Allowed',
            'details': f'Method {error.description} is not allowed for this endpoint',
            'timestamp': '2025-01-27T00:00:00Z'
        }), 405
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        """Handle request too large errors."""
        return jsonify({
            'error': 'Request Entity Too Large',
            'details': 'The uploaded file is too large',
            'timestamp': '2025-01-27T00:00:00Z'
        }), 413
    
    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        """Handle rate limit exceeded errors."""
        return jsonify({
            'error': 'Rate Limit Exceeded',
            'details': 'Too many requests. Please try again later.',
            'retry_after': getattr(error, 'retry_after', None),
            'timestamp': '2025-01-27T00:00:00Z'
        }), 429
    
    @app.errorhandler(500)
    def internal_server_error(error):
        """Handle internal server errors."""
        # Get the error message from various possible sources
        error_msg = str(error)
        
        # Check if this is actually a JWT error that should be 401
        if hasattr(error, 'original_exception'):
            original = error.original_exception
            if isinstance(original, NoAuthorizationError):
                logger.warning(f"JWT authorization error from {request.remote_addr}: {str(original)}")
                return jsonify({
                    'error': 'Unauthorized',
                    'details': 'Authorization header is required. Please provide a valid JWT token.',
                    'timestamp': '2025-01-27T00:00:00Z'
                }), 401
            elif isinstance(original, DecodeError):
                logger.warning(f"JWT decode error from {request.remote_addr}: {str(original)}")
                return jsonify({
                    'error': 'Unauthorized',
                    'details': 'Invalid token format.',
                    'timestamp': '2025-01-27T00:00:00Z'
                }), 401
            error_msg = str(original)
        
        # Check if error has a description attribute
        if hasattr(error, 'description'):
            error_msg = str(error.description)
        
        
        # Check the error message for JWT-related content
        jwt_auth_patterns = [
            'Missing Authorization Header',
            'Missing \'Bearer\' type',
            'Authorization header is required'
        ]
        
        jwt_decode_patterns = [
            'Not enough segments',
            'Invalid token',
            'Token is malformed',
            'DecodeError'
        ]
        
        # Check for authorization errors
        for pattern in jwt_auth_patterns:
            if pattern in error_msg:
                logger.warning(f"JWT authorization error from {request.remote_addr}: {error_msg}")
                return jsonify({
                    'error': 'Unauthorized',
                    'details': 'Authorization header is required. Please provide a valid JWT token.',
                    'timestamp': '2025-01-27T00:00:00Z'
                }), 401
        
        # Check for token decode errors
        for pattern in jwt_decode_patterns:
            if pattern in error_msg:
                logger.warning(f"JWT decode error from {request.remote_addr}: {error_msg}")
                return jsonify({
                    'error': 'Unauthorized',
                    'details': 'Invalid token format.',
                    'timestamp': '2025-01-27T00:00:00Z'
                }), 401
        
        logger.error(f"Internal server error: {str(error)}")
        return jsonify({
            'error': 'Internal Server Error',
            'details': 'An unexpected error occurred',
            'timestamp': '2025-01-27T00:00:00Z'
        }), 500
    
    logger.info("Error handlers registered successfully")


def register_health_check(app: Flask):
    """Register health check endpoint."""
    
    @app.route('/health')
    def health_check():
        """Health check endpoint."""
        try:
            # Check upload directory
            upload_folder = app.config['UPLOAD_FOLDER']
            upload_accessible = os.path.exists(upload_folder) and os.access(upload_folder, os.W_OK)
            
            # Check rate limiter
            rate_limiter_status = 'healthy'
            try:
                if rate_limit_manager.limiter:
                    # Try to get current limits
                    limits = rate_limit_manager.get_limits()
                    if not limits:
                        rate_limiter_status = 'degraded'
            except Exception:
                rate_limiter_status = 'unhealthy'
            
            # Overall health status
            overall_status = 'healthy'
            if not upload_accessible or rate_limiter_status == 'unhealthy':
                overall_status = 'unhealthy'
            elif rate_limiter_status == 'degraded':
                overall_status = 'degraded'
            
            health_data = {
                'status': overall_status,
                'timestamp': '2025-01-27T00:00:00Z',
                'version': '1.0.0',
                'components': {
                    'upload_directory': 'healthy' if upload_accessible else 'unhealthy',
                    'rate_limiter': rate_limiter_status,
                    'authentication': 'healthy'
                }
            }
            
            status_code = 200 if overall_status == 'healthy' else 503
            return jsonify(health_data), status_code
            
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            return jsonify({
                'status': 'unhealthy',
                'timestamp': '2025-01-27T00:00:00Z',
                'error': 'Health check failed'
            }), 503
    
    @app.route('/api/v1/health')
    def api_health_check():
        """API-specific health check."""
        return health_check()
    
    logger.info("Health check endpoints registered")


def create_development_app() -> Flask:
    """Create development application."""
    return create_api_app('development')


def create_production_app() -> Flask:
    """Create production application."""
    return create_api_app('production')


def create_testing_app() -> Flask:
    """Create testing application."""
    return create_api_app('testing')


if __name__ == '__main__':
    # Run development server
    app = create_development_app()
    
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', '5000'))
    debug = os.getenv('DEBUG', 'true').lower() == 'true'
    
    logger.info(f"Starting API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)