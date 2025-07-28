#!/usr/bin/env python3
"""
Web application entry point that works with both refactored and legacy code.
"""

import os
import sys
import json
import time
from pathlib import Path

# Try to import Flask, provide fallback if not available
try:
    from flask import Flask, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    print("⚠️  Flask not available. Please install with: pip install flask")
    FLASK_AVAILABLE = False

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Try to import refactored components
REFACTORED_AVAILABLE = False
try:
    from src.web.api_app import create_development_app, create_production_app
    from src.infrastructure.logging import get_logger
    REFACTORED_AVAILABLE = True
    logger = get_logger('web_app')
    print("✅ Refactored web modules loaded successfully")
except ImportError as e:
    print(f"⚠️  Could not load refactored web modules: {e}")
    print("🔄 Creating hybrid web application...")


def create_app(config_name: str = None):
    """
    Create Flask application with hybrid approach.
    """
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'development')
    
    if REFACTORED_AVAILABLE:
        print(f"🚀 Using refactored web application (config: {config_name})")
        if config_name == 'production':
            app = create_production_app()
        else:
            app = create_development_app()
        
        # Add backward compatibility routes
        add_compatibility_routes(app)
        return app
    else:
        print(f"🔧 Creating hybrid web application (config: {config_name})")
        return create_hybrid_app(config_name)


def create_hybrid_app(config_name: str = 'development'):
    """
    Create a hybrid Flask application that works with existing code.
    """
    if not FLASK_AVAILABLE:
        print("❌ Cannot create web application - Flask not installed")
        print("💡 Install Flask with: pip install flask")
        return None
    
    app = Flask(__name__)
    
    # Basic configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['DEBUG'] = config_name == 'development'
    app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
    
    # Create upload directory
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Basic routes
    @app.route('/')
    def index():
        """Main page with API information."""
        return jsonify({
            'message': 'File Processing Web Application',
            'version': '2.0.0',
            'mode': 'hybrid',
            'status': 'running',
            'endpoints': {
                'health': '/health',
                'status': '/status',
                'process': '/api/process',
                'files': '/api/files'
            },
            'note': 'Hybrid mode - using existing main.py functionality'
        })
    
    @app.route('/health')
    def health():
        """Health check endpoint."""
        try:
            # Check upload directory
            upload_accessible = os.path.exists(app.config['UPLOAD_FOLDER']) and os.access(app.config['UPLOAD_FOLDER'], os.W_OK)
            
            # Check if main processing is available
            main_available = False
            try:
                from main import MainApplication
                main_available = True
            except ImportError:
                pass
            
            status = 'healthy' if upload_accessible and main_available else 'degraded'
            
            return jsonify({
                'status': status,
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'version': '2.0.0',
                'mode': 'hybrid',
                'components': {
                    'upload_directory': 'healthy' if upload_accessible else 'unhealthy',
                    'main_processing': 'healthy' if main_available else 'unhealthy'
                }
            }), 200 if status == 'healthy' else 503
            
        except Exception as e:
            return jsonify({
                'status': 'unhealthy',
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'error': str(e)
            }), 503
    
    @app.route('/status')
    def status():
        """Status endpoint for backward compatibility."""
        return jsonify({
            'status': 'running',
            'mode': 'hybrid',
            'version': '2.0.0',
            'refactored_available': REFACTORED_AVAILABLE,
            'main_available': check_main_available(),
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        })
    
    @app.route('/api/process', methods=['POST'])
    def process_files():
        """Process files using main.py functionality."""
        try:
            # Get configuration from request
            config_data = request.get_json()
            if not config_data:
                return jsonify({'error': 'Configuration data required'}), 400
            
            # Try to use MainApplication from main.py
            try:
                from main import MainApplication
                app_instance = MainApplication()
                
                # Create a simple progress callback
                progress_data = {'status': 'starting', 'progress': 0, 'message': 'Initializing...'}
                
                def progress_callback(status, progress, message):
                    progress_data.update({
                        'status': status,
                        'progress': progress,
                        'message': message
                    })
                
                # Run processing
                app_instance.run_processing_optimized(config_data, progress_callback)
                
                return jsonify({
                    'success': True,
                    'message': 'Processing completed successfully',
                    'final_status': progress_data
                })
                
            except ImportError:
                return jsonify({
                    'error': 'Main processing engine not available',
                    'details': 'Could not import MainApplication from main.py'
                }), 500
                
        except Exception as e:
            return jsonify({
                'error': 'Processing failed',
                'details': str(e)
            }), 500
    
    @app.route('/api/files', methods=['GET'])
    def list_files():
        """List available files."""
        try:
            from main import MainApplication
            app_instance = MainApplication()
            
            # This would need to be implemented in MainApplication
            # For now, return a simple file listing
            files = []
            data_dir = Path('data')
            if data_dir.exists():
                for file_path in data_dir.rglob('*'):
                    if file_path.is_file() and file_path.suffix in ['.csv', '.json']:
                        files.append({
                            'name': file_path.name,
                            'path': str(file_path),
                            'size': file_path.stat().st_size,
                            'type': file_path.suffix[1:]  # Remove the dot
                        })
            
            return jsonify({
                'files': files,
                'count': len(files)
            })
            
        except Exception as e:
            return jsonify({
                'error': 'Failed to list files',
                'details': str(e)
            }), 500
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Not Found',
            'message': 'The requested endpoint was not found',
            'available_endpoints': [
                '/', '/health', '/status', '/api/process', '/api/files'
            ]
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred'
        }), 500
    
    return app


def add_compatibility_routes(app):
    """Add backward compatibility routes to refactored app."""
    @app.route('/status')
    def legacy_status():
        return jsonify({
            'status': 'running',
            'mode': 'refactored',
            'version': '2.0.0',
            'api_documentation': '/api/v1/docs/',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        })


def check_main_available():
    """Check if main.py functionality is available."""
    try:
        from main import MainApplication
        return True
    except ImportError:
        return False


def main():
    """Main entry point."""
    config_name = os.getenv('FLASK_ENV', 'development')
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', '5000'))
    debug = os.getenv('DEBUG', 'true').lower() == 'true' and config_name == 'development'
    
    print("🌐 Starting File Processing Web Application")
    print("=" * 50)
    print(f"📍 Server: http://{host}:{port}")
    print(f"🔧 Configuration: {config_name}")
    print(f"🐛 Debug mode: {debug}")
    print(f"📁 Upload folder: {os.getenv('UPLOAD_FOLDER', 'uploads')}")
    
    if REFACTORED_AVAILABLE:
        print(f"📚 API Documentation: http://{host}:{port}/api/v1/docs/")
    else:
        print(f"📋 API Endpoints: http://{host}:{port}/")
    
    print("=" * 50)
    
    app = create_app(config_name)
    
    try:
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n👋 Web application stopped by user")
    except Exception as e:
        print(f"❌ Web application error: {e}")
        raise


# WSGI application for production deployment
def create_wsgi_app():
    """Create WSGI application."""
    config_name = os.getenv('FLASK_ENV', 'production')
    return create_app(config_name)


# Application instance for WSGI servers
application = create_wsgi_app()


if __name__ == '__main__':
    main()