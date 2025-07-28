"""
REST API routes for file processing operations.
Implements requirements 6.1, 6.4, 7.2, 4.1: REST API endpoints with authentication and comprehensive documentation.
"""

import os
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from flask import Blueprint, request, jsonify, current_app, send_file
from flask_restx import Api, Resource, fields, Namespace
from flask_jwt_extended import (
    jwt_required, create_access_token, get_jwt_identity, 
    get_jwt, verify_jwt_in_request
)
from flask_jwt_extended.exceptions import NoAuthorizationError
from werkzeug.exceptions import BadRequest, NotFound, Unauthorized, Forbidden
from werkzeug.datastructures import FileStorage

from src.web.services.file_service import FileService, SessionService
from src.web.services.processing_service import ProcessingService
from src.web.services.websocket_progress_service import websocket_progress_service
from src.web.services.job_service import job_service, JobStatus, JobPriority
from src.web.models.web_models import (
    FileUpload, ProcessingConfig, FieldMapping, SessionData
)
from src.domain.exceptions import FileProcessingError, FileValidationError
from src.infrastructure.logging import get_logger

# Create blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

# Initialize Flask-RESTX
api = Api(
    api_bp,
    version='1.0',
    title='File Processing API',
    description='REST API for file processing and data matching operations',
    doc='/docs/',
    authorizations={
        'Bearer': {
            'type': 'apiKey',
            'in': 'header',
            'name': 'Authorization',
            'description': 'JWT token in format: Bearer <token>'
        }
    },
    security='Bearer'
)

# Initialize services
file_service = FileService(upload_folder='uploads')
session_service = SessionService()
processing_service = ProcessingService()
logger = get_logger('api')

# Namespaces
auth_ns = Namespace('auth', description='Authentication operations')
files_ns = Namespace('files', description='File upload and validation operations')
processing_ns = Namespace('processing', description='Data processing operations')
results_ns = Namespace('results', description='Result management operations')
jobs_ns = Namespace('jobs', description='Job queue and batch processing operations')

api.add_namespace(auth_ns)
api.add_namespace(files_ns)
api.add_namespace(processing_ns)
api.add_namespace(results_ns)
api.add_namespace(jobs_ns)

# API Models for documentation
auth_model = api.model('AuthRequest', {
    'username': fields.String(required=True, description='Username'),
    'password': fields.String(required=True, description='Password')
})

auth_response_model = api.model('AuthResponse', {
    'access_token': fields.String(description='JWT access token'),
    'expires_in': fields.Integer(description='Token expiration time in seconds')
})

file_upload_model = api.model('FileUpload', {
    'original_name': fields.String(description='Original filename'),
    'unique_name': fields.String(description='Unique filename'),
    'file_type': fields.String(description='File type (csv/json)'),
    'delimiter': fields.String(description='CSV delimiter (if applicable)'),
    'upload_timestamp': fields.DateTime(description='Upload timestamp')
})

field_mapping_model = api.model('FieldMapping', {
    'file1_col': fields.String(required=True, description='Column from first file'),
    'file2_col': fields.String(required=True, description='Column from second file'),
    'match_type': fields.String(description='Matching algorithm type', enum=['exact', 'fuzzy', 'phonetic']),
    'use_normalization': fields.Boolean(description='Use text normalization'),
    'case_sensitive': fields.Boolean(description='Case sensitive matching'),
    'weight': fields.Float(description='Field weight for matching')
})

processing_config_model = api.model('ProcessingConfig', {
    'mappings': fields.List(fields.Nested(field_mapping_model), required=True),
    'output_cols1': fields.List(fields.String, description='Output columns from file 1'),
    'output_cols2': fields.List(fields.String, description='Output columns from file 2'),
    'output_format': fields.String(description='Output format', enum=['json', 'csv']),
    'output_path': fields.String(description='Output file path'),
    'threshold': fields.Integer(description='Confidence threshold (0-100)'),
    'matching_type': fields.String(description='Matching type', enum=['one-to-one', 'one-to-many']),
    'generate_unmatched': fields.Boolean(description='Generate unmatched records files')
})

progress_status_model = api.model('ProgressStatus', {
    'operation_id': fields.String(description='Operation identifier'),
    'status': fields.String(description='Current status', enum=['idle', 'starting', 'processing', 'completed', 'error', 'cancelled']),
    'progress': fields.Integer(description='Progress percentage (0-100)'),
    'message': fields.String(description='Status message'),
    'elapsed_time': fields.Integer(description='Elapsed time in seconds'),
    'estimated_remaining': fields.Integer(description='Estimated remaining time in seconds'),
    'can_cancel': fields.Boolean(description='Whether operation can be cancelled')
})

result_file_model = api.model('ResultFile', {
    'name': fields.String(description='Display name'),
    'path': fields.String(description='File path'),
    'type': fields.String(description='File type', enum=['matched', 'low_confidence', 'unmatched_1', 'unmatched_2']),
    'count': fields.Integer(description='Number of records'),
    'columns': fields.List(fields.String, description='Column names')
})

error_model = api.model('Error', {
    'error': fields.String(description='Error message'),
    'details': fields.String(description='Detailed error information'),
    'timestamp': fields.DateTime(description='Error timestamp')
})

# Job management models
job_model = api.model('Job', {
    'id': fields.String(description='Job identifier'),
    'name': fields.String(description='Job name'),
    'status': fields.String(description='Job status', enum=['pending', 'running', 'completed', 'failed', 'cancelled']),
    'priority': fields.Integer(description='Job priority (1-4)'),
    'created_at': fields.DateTime(description='Job creation timestamp'),
    'started_at': fields.DateTime(description='Job start timestamp'),
    'completed_at': fields.DateTime(description='Job completion timestamp'),
    'progress': fields.Integer(description='Progress percentage (0-100)'),
    'message': fields.String(description='Status message'),
    'error_message': fields.String(description='Error message if failed'),
    'result_files': fields.List(fields.String, description='Result file paths'),
    'user_id': fields.String(description='User who submitted the job'),
    'duration': fields.Integer(description='Job duration in seconds'),
    'estimated_remaining': fields.Integer(description='Estimated remaining time in seconds')
})

job_submit_model = api.model('JobSubmit', {
    'name': fields.String(required=True, description='Job name'),
    'config': fields.Nested(processing_config_model, required=True, description='Processing configuration'),
    'priority': fields.Integer(description='Job priority (1=low, 2=normal, 3=high, 4=urgent)', default=2)
})

job_stats_model = api.model('JobStats', {
    'total_jobs': fields.Integer(description='Total number of jobs'),
    'pending': fields.Integer(description='Number of pending jobs'),
    'running': fields.Integer(description='Number of running jobs'),
    'completed': fields.Integer(description='Number of completed jobs'),
    'failed': fields.Integer(description='Number of failed jobs'),
    'cancelled': fields.Integer(description='Number of cancelled jobs'),
    'active_workers': fields.Integer(description='Number of active workers'),
    'total_workers': fields.Integer(description='Total number of workers')
})

# Authentication endpoints
@auth_ns.route('/login')
class AuthLogin(Resource):
    @auth_ns.expect(auth_model)
    @auth_ns.marshal_with(auth_response_model)
    @auth_ns.doc('authenticate_user')
    def post(self):
        """Authenticate user and return JWT token"""
        try:
            data = request.get_json()
            username = data.get('username')
            password = data.get('password')
            
            # Simple authentication (in production, use proper user management)
            if not username or not password:
                raise BadRequest("Username and password are required")
            
            # For demo purposes, accept any non-empty credentials
            # In production, validate against user database
            if len(username.strip()) == 0 or len(password.strip()) == 0:
                raise Unauthorized("Invalid credentials")
            
            # Create JWT token
            expires = timedelta(hours=24)
            access_token = create_access_token(
                identity=username,
                expires_delta=expires
            )
            
            logger.info(f"User {username} authenticated successfully")
            
            return {
                'access_token': access_token,
                'expires_in': int(expires.total_seconds())
            }
            
        except BadRequest as e:
            raise e
        except Unauthorized as e:
            raise e
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            raise BadRequest(f"Authentication failed: {str(e)}")

@auth_ns.route('/verify')
class AuthVerify(Resource):
    @jwt_required()
    @auth_ns.doc('verify_token')
    def get(self):
        """Verify JWT token validity"""
        try:
            current_user = get_jwt_identity()
            jwt_data = get_jwt()
            
            return {
                'valid': True,
                'user': current_user,
                'expires_at': jwt_data.get('exp')
            }
            
        except Exception as e:
            logger.error(f"Token verification error: {str(e)}")
            raise Unauthorized("Invalid or expired token")

# File management endpoints
@files_ns.route('/upload')
class FileUpload(Resource):
    @jwt_required()
    @files_ns.doc('upload_files')
    @files_ns.marshal_with(file_upload_model, as_list=True)
    def post(self):
        """Upload files for processing"""
        try:
            if 'file1' not in request.files or 'file2' not in request.files:
                raise BadRequest("Both file1 and file2 are required")
            
            file1 = request.files['file1']
            file2 = request.files['file2']
            
            if not file1.filename or not file2.filename:
                raise BadRequest("Both files must have valid filenames")
            
            # Save uploaded files
            file1_upload = file_service.save_uploaded_file(file1)
            file2_upload = file_service.save_uploaded_file(file2)
            
            # Create session data
            session_data = SessionData(
                file1=file1_upload,
                file2=file2_upload,
                created_at=datetime.now()
            )
            session_service.save_session(session_data)
            
            logger.info(f"Files uploaded successfully: {file1_upload.original_name}, {file2_upload.original_name}")
            
            return [file1_upload.to_dict(), file2_upload.to_dict()]
            
        except FileValidationError as e:
            raise BadRequest(str(e))
        except FileProcessingError as e:
            raise BadRequest(str(e))
        except Exception as e:
            logger.error(f"File upload error: {str(e)}")
            raise BadRequest(f"File upload failed: {str(e)}")

@files_ns.route('/validate')
class FileValidation(Resource):
    @jwt_required()
    @files_ns.doc('validate_files')
    def get(self):
        """Validate uploaded files and return structure information"""
        try:
            session_data = session_service.load_session()
            if not session_data or not session_data.file1 or not session_data.file2:
                raise NotFound("No uploaded files found. Please upload files first.")
            
            # Validate both files
            validation1 = file_service.validate_file_structure(session_data.file1)
            validation2 = file_service.validate_file_structure(session_data.file2)
            
            # Get file previews
            preview1 = file_service.get_file_preview(session_data.file1)
            preview2 = file_service.get_file_preview(session_data.file2)
            
            return {
                'file1': {
                    'validation': validation1,
                    'preview': preview1
                },
                'file2': {
                    'validation': validation2,
                    'preview': preview2
                }
            }
            
        except FileProcessingError as e:
            raise BadRequest(str(e))
        except Exception as e:
            logger.error(f"File validation error: {str(e)}")
            raise BadRequest(f"File validation failed: {str(e)}")

@files_ns.route('/preview')
class FilePreview(Resource):
    @jwt_required()
    @files_ns.doc('get_file_preview')
    def get(self):
        """Get preview of uploaded files"""
        try:
            session_data = session_service.load_session()
            if not session_data or not session_data.file1 or not session_data.file2:
                raise NotFound("No uploaded files found. Please upload files first.")
            
            rows = request.args.get('rows', 5, type=int)
            rows = max(1, min(100, rows))  # Limit between 1 and 100
            
            preview1 = file_service.get_file_preview(session_data.file1, rows)
            preview2 = file_service.get_file_preview(session_data.file2, rows)
            
            return {
                'file1': preview1,
                'file2': preview2
            }
            
        except FileProcessingError as e:
            raise BadRequest(str(e))
        except NoAuthorizationError:
            # Let JWT errors pass through to be handled by Flask error handlers
            raise
        except Exception as e:
            logger.error(f"File preview error: {str(e)}")
            raise BadRequest(f"File preview failed: {str(e)}")

# Processing endpoints
@processing_ns.route('/start')
class ProcessingStart(Resource):
    @jwt_required()
    @processing_ns.expect(processing_config_model)
    @processing_ns.marshal_with(progress_status_model)
    @processing_ns.doc('start_processing')
    def post(self):
        """Start file processing operation"""
        try:
            session_data = session_service.load_session()
            if not session_data or not session_data.file1 or not session_data.file2:
                raise NotFound("No uploaded files found. Please upload files first.")
            
            config_data = request.get_json()
            if not config_data:
                raise BadRequest("Processing configuration is required")
            
            # Validate required fields
            if 'mappings' not in config_data or not config_data['mappings']:
                raise BadRequest("At least one field mapping is required")
            
            # Create field mappings
            mappings = []
            for mapping_data in config_data['mappings']:
                if 'file1_col' not in mapping_data or 'file2_col' not in mapping_data:
                    raise BadRequest("Each mapping must specify file1_col and file2_col")
                mappings.append(FieldMapping.from_dict(mapping_data))
            
            # Output columns - agar bo'sh bo'lsa barcha maydonlarni olish
            output_cols1 = config_data.get('output_cols1', [])
            output_cols2 = config_data.get('output_cols2', [])
            
            # Agar output_cols1 bo'sh bo'lsa, file1 dan barcha maydonlarni olish
            if not output_cols1:
                try:
                    validation1 = file_service.validate_file_structure(session_data.file1)
                    if validation1.get('valid') and validation1.get('info', {}).get('column_names'):
                        output_cols1 = validation1['info']['column_names']
                except Exception:
                    output_cols1 = []
            
            # Agar output_cols2 bo'sh bo'lsa, file2 dan barcha maydonlarni olish
            if not output_cols2:
                try:
                    validation2 = file_service.validate_file_structure(session_data.file2)
                    if validation2.get('valid') and validation2.get('info', {}).get('column_names'):
                        output_cols2 = validation2['info']['column_names']
                except Exception:
                    output_cols2 = []
            
            # Create processing configuration
            processing_config = ProcessingConfig(
                file1=session_data.file1,
                file2=session_data.file2,
                mappings=mappings,
                output_cols1=output_cols1,
                output_cols2=output_cols2,
                output_format=config_data.get('output_format', 'json'),
                output_path=config_data.get('output_path', f'matched_results_{uuid.uuid4().hex[:8]}'),
                prefix1=config_data.get('prefix1', 'f1_'),
                prefix2=config_data.get('prefix2', 'f2_'),
                threshold=config_data.get('threshold', 75),
                matching_type=config_data.get('matching_type', 'one-to-one'),
                generate_unmatched=config_data.get('generate_unmatched', False)
            )
            
            # Start processing
            success = processing_service.start_processing(processing_config)
            if not success:
                raise BadRequest("Failed to start processing. Another operation may be in progress.")
            
            # Get initial progress status
            progress = processing_service.get_progress()

            logger.info(f"Processing started for user {get_jwt_identity()}")

            result = progress.to_dict()

            return result
            
        except FileProcessingError as e:
            raise BadRequest(str(e))
        except Exception as e:
            logger.error(f"Processing start error: {str(e)}")
            raise BadRequest(f"Failed to start processing: {str(e)}")

@processing_ns.route('/status')
class ProcessingStatus(Resource):
    @jwt_required()
    @processing_ns.marshal_with(progress_status_model)
    @processing_ns.doc('get_processing_status')
    def get(self):
        """Get current processing status"""
        try:
            progress = processing_service.get_progress()
            return progress.to_dict()
            
        except Exception as e:
            logger.error(f"Status retrieval error: {str(e)}")
            raise BadRequest(f"Failed to get status: {str(e)}")

@processing_ns.route('/cancel')
class ProcessingCancel(Resource):
    @jwt_required()
    @processing_ns.doc('cancel_processing')
    def post(self):
        """Cancel current processing operation"""
        try:
            # For now, we don't have direct cancellation in processing_service
            # This would need to be implemented with proper cancellation tokens
            
            return {
                'success': False,
                'message': 'Cancellation not yet implemented in processing service'
            }
            
        except Exception as e:
            logger.error(f"Processing cancellation error: {str(e)}")
            raise BadRequest(f"Failed to cancel processing: {str(e)}")

# Results endpoints
@results_ns.route('/files')
class ResultFiles(Resource):
    @jwt_required()
    @results_ns.marshal_with(result_file_model, as_list=True)
    @results_ns.doc('get_result_files')
    def get(self):
        """Get list of result files"""
        try:
            result_files = processing_service.get_result_files()
            return [rf.to_dict() for rf in result_files]
            
        except Exception as e:
            logger.error(f"Result files retrieval error: {str(e)}")
            raise BadRequest(f"Failed to get result files: {str(e)}")

@results_ns.route('/data/<string:file_type>')
class ResultData(Resource):
    @jwt_required()
    @results_ns.doc('get_result_data', params={
        'page': 'Page number (default: 1)',
        'per_page': 'Items per page (default: 50, max: 1000)',
        'search': 'Search term to filter results',
        'confidence_threshold': 'Confidence threshold for sorting (default: 80.0). High confidence (>=threshold) sorted ascending, low confidence (<threshold) sorted descending'
    })
    def get(self, file_type):
        """Get paginated result data with confidence score sorting
        
        Results are automatically sorted by confidence score:
        - High confidence records (>=confidence_threshold): ascending order
        - Low confidence records (<confidence_threshold): descending order
        """
        try:
            if file_type not in ['matched', 'low_confidence', 'unmatched_1', 'unmatched_2']:
                raise BadRequest("Invalid file type")
            
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 50, type=int)
            search = request.args.get('search', '', type=str)
            confidence_threshold = request.args.get('confidence_threshold', 80.0, type=float)
            
            # Limit per_page to reasonable values
            per_page = max(1, min(1000, per_page))
            page = max(1, page)
            confidence_threshold = max(0.0, min(100.0, confidence_threshold))
            
            data = processing_service.get_paginated_data(
                file_type=file_type,
                page=page,
                per_page=per_page,
                search=search,
                confidence_threshold=confidence_threshold
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Result data retrieval error: {str(e)}")
            raise BadRequest(f"Failed to get result data: {str(e)}")

@results_ns.route('/download/<string:file_type>')
class ResultDownload(Resource):
    @jwt_required()
    @results_ns.doc('download_result_file')
    def get(self, file_type):
        """Download result file with optional format conversion"""
        try:
            import os
            import pandas as pd
            import tempfile
            import json
            
            if file_type not in ['matched', 'low_confidence', 'unmatched_1', 'unmatched_2']:
                raise BadRequest("Invalid file type")
            
            # Format parametrini olish (json yoki csv)
            requested_format = request.args.get('format', '').lower()
            if requested_format and requested_format not in ['json', 'csv']:
                raise BadRequest("Invalid format. Supported formats: json, csv")
            
            # Original faylni topish
            result_files = processing_service.get_result_files()
            target_file = None
            for rf in result_files:
                if rf.file_type == file_type:
                    target_file = rf
                    break
            
            if not target_file or not os.path.exists(target_file.path):
                raise NotFound(f"Result file not found: {file_type}")
            
            # Agar format ko'rsatilmagan bo'lsa, original faylni qaytarish
            if not requested_format:
                return send_file(
                    target_file.path,
                    as_attachment=True,
                    download_name=f"{target_file.name.lower().replace(' ', '_')}.{target_file.path.split('.')[-1]}"
                )
            
            # Format conversion kerak bo'lsa
            original_format = target_file.path.split('.')[-1].lower()
            
            # Agar original format va requested format bir xil bo'lsa
            if original_format == requested_format:
                return send_file(
                    target_file.path,
                    as_attachment=True,
                    download_name=f"{target_file.name.lower().replace(' ', '_')}.{requested_format}"
                )
            
            # Format conversion
            import pandas as pd
            import tempfile
            import os
            
            # Original faylni o'qish
            if original_format == 'json':
                try:
                    df = pd.read_json(target_file.path, lines=True)
                except:
                    try:
                        df = pd.read_json(target_file.path)
                    except:
                        import json
                        with open(target_file.path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            df = pd.DataFrame(data)
                        elif isinstance(data, dict):
                            df = pd.DataFrame([data])
                        else:
                            df = pd.DataFrame()
            else:  # CSV
                df = pd.read_csv(target_file.path)
            
            # Temporary fayl yaratish
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=f'.{requested_format}',
                prefix=f'{file_type}_'
            )
            temp_file.close()
            
            # Kerakli formatda saqlash
            if requested_format == 'json':
                df.to_json(temp_file.name, orient='records', indent=2, ensure_ascii=False)
            else:  # CSV
                # CSV export-ni yaxshilash
                df.to_csv(
                    temp_file.name, 
                    index=False, 
                    encoding='utf-8-sig',  # BOM bilan UTF-8 (Excel uchun yaxshi)
                    sep=',',  # Comma separator
                    quoting=1,  # Quote non-numeric fields
                    escapechar='\\',  # Escape character
                    na_rep='',  # Empty string for NaN values
                    float_format='%.2f'  # 2 decimal places for floats
                )
            
            # Faylni qaytarish va keyin o'chirish
            def remove_file(response):
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
                return response
            
            response = send_file(
                temp_file.name,
                as_attachment=True,
                download_name=f"{target_file.name.lower().replace(' ', '_')}.{requested_format}"
            )
            
            # Temporary faylni o'chirish uchun callback qo'shish
            response.call_on_close(lambda: os.unlink(temp_file.name) if os.path.exists(temp_file.name) else None)
            
            return response
            
        except NotFound as e:
            raise e
        except Exception as e:
            logger.error(f"Result download error: {str(e)}")
            raise BadRequest(f"Failed to download result: {str(e)}")

@results_ns.route('/cleanup')
class ResultCleanup(Resource):
    @jwt_required()
    @results_ns.doc('cleanup_results')
    def delete(self):
        """Clean up result files and session data"""
        try:
            # Clean up result files
            removed_results = processing_service.cleanup_results()
            
            # Clean up uploaded files
            session_data = session_service.load_session()
            removed_uploads = 0
            
            if session_data:
                files_to_cleanup = []
                if session_data.file1:
                    files_to_cleanup.append(session_data.file1)
                if session_data.file2:
                    files_to_cleanup.append(session_data.file2)
                
                removed_uploads = file_service.cleanup_files(files_to_cleanup)
                
                # Clear session
                session_service.clear_session()
            
            logger.info(f"Cleanup completed: {removed_results} result files, {removed_uploads} upload files")
            
            return {
                'success': True,
                'removed_result_files': removed_results,
                'removed_upload_files': removed_uploads,
                'message': f'Cleaned up {removed_results + removed_uploads} files'
            }
            
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")
            raise BadRequest(f"Cleanup failed: {str(e)}")

# Job management endpoints
@jobs_ns.route('/submit')
class JobSubmit(Resource):
    @jwt_required()
    @jobs_ns.expect(job_submit_model)
    @jobs_ns.doc('submit_job')
    def post(self):
        """Submit a new job for batch processing"""
        try:
            data = request.get_json()
            if not data:
                raise BadRequest("Job data is required")
            
            name = data.get('name')
            config = data.get('config')
            priority = data.get('priority', 2)
            
            if not name:
                raise BadRequest("Job name is required")
            
            if not config:
                raise BadRequest("Job configuration is required")
            
            # Validate priority
            if priority not in [1, 2, 3, 4]:
                raise BadRequest("Priority must be between 1 (low) and 4 (urgent)")
            
            # Convert priority to enum
            priority_enum = JobPriority(priority)
            
            # Get current user
            user_id = get_jwt_identity()
            
            # Submit job
            job_id = job_service.submit_job(
                name=name,
                config=config,
                priority=priority_enum,
                user_id=user_id
            )
            
            logger.info(f"Job {job_id} submitted by user {user_id}")
            
            return {
                'job_id': job_id,
                'message': 'Job submitted successfully'
            }
            
        except Exception as e:
            logger.error(f"Job submission error: {str(e)}")
            raise BadRequest(f"Failed to submit job: {str(e)}")

@jobs_ns.route('/list')
class JobList(Resource):
    @jwt_required()
    @jobs_ns.marshal_with(job_model, as_list=True)
    @jobs_ns.doc('list_jobs')
    def get(self):
        """List jobs with optional filtering"""
        try:
            # Get query parameters
            status = request.args.get('status')
            limit = request.args.get('limit', type=int)
            user_id = get_jwt_identity()  # Only show user's own jobs
            
            # Convert status string to enum
            status_enum = None
            if status:
                try:
                    status_enum = JobStatus(status.lower())
                except ValueError:
                    raise BadRequest(f"Invalid status: {status}")
            
            # Get jobs
            jobs = job_service.list_jobs(
                user_id=user_id,
                status=status_enum,
                limit=limit
            )
            
            return [job.to_dict() for job in jobs]
            
        except Exception as e:
            logger.error(f"Job listing error: {str(e)}")
            raise BadRequest(f"Failed to list jobs: {str(e)}")

@jobs_ns.route('/<string:job_id>')
class JobDetail(Resource):
    @jwt_required()
    @jobs_ns.marshal_with(job_model)
    @jobs_ns.doc('get_job')
    def get(self, job_id):
        """Get job details by ID"""
        try:
            job = job_service.get_job(job_id)
            if not job:
                raise NotFound(f"Job not found: {job_id}")
            
            # Check if user owns this job
            current_user = get_jwt_identity()
            if job.user_id != current_user:
                raise Forbidden("Access denied to this job")
            
            return job.to_dict()
            
        except NotFound as e:
            raise e
        except Forbidden as e:
            raise e
        except Exception as e:
            logger.error(f"Job retrieval error: {str(e)}")
            raise BadRequest(f"Failed to get job: {str(e)}")
    
    @jwt_required()
    @jobs_ns.doc('delete_job')
    def delete(self, job_id):
        """Delete a completed job"""
        try:
            job = job_service.get_job(job_id)
            if not job:
                raise NotFound(f"Job not found: {job_id}")
            
            # Check if user owns this job
            current_user = get_jwt_identity()
            if job.user_id != current_user:
                raise Forbidden("Access denied to this job")
            
            # Delete job
            success = job_service.delete_job(job_id)
            if not success:
                raise BadRequest("Cannot delete job (may be running or pending)")
            
            logger.info(f"Job {job_id} deleted by user {current_user}")
            
            return {
                'success': True,
                'message': 'Job deleted successfully'
            }
            
        except NotFound as e:
            raise e
        except Forbidden as e:
            raise e
        except Exception as e:
            logger.error(f"Job deletion error: {str(e)}")
            raise BadRequest(f"Failed to delete job: {str(e)}")

@jobs_ns.route('/<string:job_id>/cancel')
class JobCancel(Resource):
    @jwt_required()
    @jobs_ns.doc('cancel_job')
    def post(self, job_id):
        """Cancel a running or pending job"""
        try:
            job = job_service.get_job(job_id)
            if not job:
                raise NotFound(f"Job not found: {job_id}")
            
            # Check if user owns this job
            current_user = get_jwt_identity()
            if job.user_id != current_user:
                raise Forbidden("Access denied to this job")
            
            # Cancel job
            success = job_service.cancel_job(job_id)
            if not success:
                raise BadRequest("Cannot cancel job (may already be completed or cancelled)")
            
            logger.info(f"Job {job_id} cancelled by user {current_user}")
            
            return {
                'success': True,
                'message': 'Job cancelled successfully'
            }
            
        except NotFound as e:
            raise e
        except Forbidden as e:
            raise e
        except Exception as e:
            logger.error(f"Job cancellation error: {str(e)}")
            raise BadRequest(f"Failed to cancel job: {str(e)}")

@jobs_ns.route('/stats')
class JobStats(Resource):
    @jwt_required()
    @jobs_ns.marshal_with(job_stats_model)
    @jobs_ns.doc('get_job_stats')
    def get(self):
        """Get job queue statistics"""
        try:
            stats = job_service.get_queue_stats()
            return stats
            
        except Exception as e:
            logger.error(f"Job stats error: {str(e)}")
            raise BadRequest(f"Failed to get job stats: {str(e)}")

@jobs_ns.route('/cleanup')
class JobCleanup(Resource):
    @jwt_required()
    @jobs_ns.doc('cleanup_old_jobs')
    def post(self):
        """Clean up old completed jobs"""
        try:
            max_age_hours = request.args.get('max_age_hours', 24, type=int)
            
            # Validate max_age_hours
            if max_age_hours < 1 or max_age_hours > 168:  # 1 hour to 1 week
                raise BadRequest("max_age_hours must be between 1 and 168")
            
            removed_count = job_service.cleanup_old_jobs(max_age_hours)
            
            logger.info(f"Cleaned up {removed_count} old jobs")
            
            return {
                'success': True,
                'removed_jobs': removed_count,
                'message': f'Cleaned up {removed_count} old jobs'
            }
            
        except Exception as e:
            logger.error(f"Job cleanup error: {str(e)}")
            raise BadRequest(f"Failed to cleanup jobs: {str(e)}")

# Error handlers
@api.errorhandler(BadRequest)
def handle_bad_request(error):
    """Handle bad request errors"""
    return {
        'error': 'Bad Request',
        'details': str(error.description),
        'timestamp': datetime.now().isoformat()
    }, 400

@api.errorhandler(Unauthorized)
def handle_unauthorized(error):
    """Handle unauthorized errors"""
    return {
        'error': 'Unauthorized',
        'details': str(error.description),
        'timestamp': datetime.now().isoformat()
    }, 401

@api.errorhandler(Forbidden)
def handle_forbidden(error):
    """Handle forbidden errors"""
    return {
        'error': 'Forbidden',
        'details': str(error.description),
        'timestamp': datetime.now().isoformat()
    }, 403

@api.errorhandler(NotFound)
def handle_not_found(error):
    """Handle not found errors"""
    return {
        'error': 'Not Found',
        'details': str(error.description),
        'timestamp': datetime.now().isoformat()
    }, 404

@api.errorhandler(Exception)
def handle_generic_error(error):
    """Handle generic errors"""
    error_msg = str(error)
    
    # Check if this is a JWT authentication error
    if isinstance(error, NoAuthorizationError) or 'Missing Authorization Header' in error_msg or 'Missing \'Bearer\' type' in error_msg:
        logger.warning(f"JWT authorization error: {error_msg}")
        return {
            'error': 'Unauthorized',
            'details': 'Authorization header is required. Please provide a valid JWT token.',
            'timestamp': datetime.now().isoformat()
        }, 401
    
    # Check if this is a JWT decode error
    from jwt.exceptions import DecodeError
    if isinstance(error, DecodeError) or 'Not enough segments' in error_msg or 'Invalid token' in error_msg:
        logger.warning(f"JWT decode error: {error_msg}")
        return {
            'error': 'Unauthorized',
            'details': 'Invalid token format.',
            'timestamp': datetime.now().isoformat()
        }, 401
    
    # Check if this is a content type error
    from werkzeug.exceptions import UnsupportedMediaType
    if isinstance(error, UnsupportedMediaType) or 'Content-Type was not \'application/json\'' in error_msg:
        logger.warning(f"Content type error: {error_msg}")
        return {
            'error': 'Unsupported Media Type',
            'details': 'Request must have Content-Type: application/json header for JSON endpoints.',
            'timestamp': datetime.now().isoformat()
        }, 415
    
    # Handle all other errors as 500
    logger.error(f"Unhandled API error: {str(error)}")
    return {
        'error': 'Internal Server Error',
        'details': 'An unexpected error occurred',
        'timestamp': datetime.now().isoformat()
    }, 500