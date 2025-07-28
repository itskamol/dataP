#!/usr/bin/env python3
"""
Generate OpenAPI/Swagger documentation for the REST API.
Implements requirement: Create comprehensive API documentation with OpenAPI/Swagger.
"""

import json
import yaml
import os
import sys
from datetime import datetime
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.web.api_app import create_development_app


def generate_openapi_spec() -> Dict[str, Any]:
    """Generate OpenAPI specification."""
    
    app = create_development_app()
    app.config['SERVER_NAME'] = 'localhost:5000'
    
    with app.app_context():
        # Get the API instance from the blueprint
        from src.web.routes.api_routes import api
        
        try:
            # Generate OpenAPI spec
            spec = api.__schema__
        except Exception as e:
            print(f"Error generating schema: {e}")
            # Create a basic spec manually
            spec = {
                'openapi': '3.0.0',
                'info': {
                    'title': 'File Processing API',
                    'version': '1.0.0'
                },
                'paths': {}
            }
        
        # Add additional metadata
        spec['info'].update({
            'title': 'File Processing API',
            'version': '1.0.0',
            'description': '''
# File Processing and Data Matching API

This API provides endpoints for uploading, processing, and matching data between two files (CSV or JSON format).

## Features

- **File Upload**: Upload CSV and JSON files for processing
- **Data Validation**: Validate file structure and content
- **Matching Algorithms**: Support for exact, fuzzy, and phonetic matching
- **Real-time Progress**: Track processing progress in real-time
- **Result Management**: Download and manage processing results
- **Authentication**: JWT-based authentication and authorization
- **Rate Limiting**: Built-in rate limiting for API protection

## Authentication

All API endpoints (except `/auth/login`) require JWT authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

## Rate Limits

- Default: 1000 requests per hour, 100 per minute
- Authenticated users: 2000 requests per hour, 200 per minute
- File uploads: 50 requests per hour, 10 per minute
- Processing operations: 20 requests per hour, 5 per minute

## Error Handling

The API returns structured error responses with the following format:

```json
{
  "error": "Error Type",
  "details": "Detailed error message",
  "timestamp": "2025-01-27T00:00:00Z"
}
```

## Workflow

1. **Authenticate**: Get JWT token using `/auth/login`
2. **Upload Files**: Upload two files using `/files/upload`
3. **Validate**: Validate file structure using `/files/validate`
4. **Configure**: Set up processing configuration
5. **Process**: Start processing using `/processing/start`
6. **Monitor**: Track progress using `/processing/status`
7. **Results**: Download results using `/results/download/{type}`
8. **Cleanup**: Clean up files using `/results/cleanup`
            ''',
            'contact': {
                'name': 'API Support',
                'email': 'support@example.com'
            },
            'license': {
                'name': 'MIT',
                'url': 'https://opensource.org/licenses/MIT'
            }
        })
        
        # Add servers
        spec['servers'] = [
            {
                'url': 'http://localhost:5000/api/v1',
                'description': 'Development server'
            },
            {
                'url': 'https://api.example.com/v1',
                'description': 'Production server'
            }
        ]
        
        # Add security schemes
        spec['components']['securitySchemes'] = {
            'Bearer': {
                'type': 'http',
                'scheme': 'bearer',
                'bearerFormat': 'JWT',
                'description': 'JWT token obtained from /auth/login endpoint'
            }
        }
        
        # Add global security requirement
        spec['security'] = [{'Bearer': []}]
        
        # Add tags
        spec['tags'] = [
            {
                'name': 'Authentication',
                'description': 'User authentication and token management'
            },
            {
                'name': 'File Management',
                'description': 'File upload, validation, and preview operations'
            },
            {
                'name': 'Processing',
                'description': 'Data processing and matching operations'
            },
            {
                'name': 'Results',
                'description': 'Result management and download operations'
            }
        ]
        
        return spec


def save_documentation(spec: Dict[str, Any], output_dir: str = 'docs/api'):
    """Save API documentation in multiple formats."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON
    json_path = os.path.join(output_dir, 'openapi.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
    
    # Save as YAML
    yaml_path = os.path.join(output_dir, 'openapi.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(spec, f, default_flow_style=False, allow_unicode=True)
    
    # Generate HTML documentation
    html_content = generate_html_docs(spec)
    html_path = os.path.join(output_dir, 'index.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"API documentation generated:")
    print(f"  JSON: {json_path}")
    print(f"  YAML: {yaml_path}")
    print(f"  HTML: {html_path}")


def generate_html_docs(spec: Dict[str, Any]) -> str:
    """Generate HTML documentation using Swagger UI."""
    
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - API Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui.css" />
    <style>
        html {{
            box-sizing: border-box;
            overflow: -moz-scrollbars-vertical;
            overflow-y: scroll;
        }}
        *, *:before, *:after {{
            box-sizing: inherit;
        }}
        body {{
            margin:0;
            background: #fafafa;
        }}
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {{
            const ui = SwaggerUIBundle({{
                url: './openapi.json',
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",
                validatorUrl: null,
                tryItOutEnabled: true,
                supportedSubmitMethods: ['get', 'post', 'put', 'delete', 'patch'],
                onComplete: function() {{
                    console.log('Swagger UI loaded');
                }},
                onFailure: function(data) {{
                    console.error('Failed to load API spec', data);
                }}
            }});
        }};
    </script>
</body>
</html>'''
    
    return html_template.format(
        title=spec['info']['title']
    )


def generate_postman_collection(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Generate Postman collection from OpenAPI spec."""
    
    collection = {
        "info": {
            "name": spec['info']['title'],
            "description": spec['info']['description'],
            "version": spec['info']['version'],
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
        },
        "auth": {
            "type": "bearer",
            "bearer": [
                {
                    "key": "token",
                    "value": "{{jwt_token}}",
                    "type": "string"
                }
            ]
        },
        "variable": [
            {
                "key": "base_url",
                "value": "http://localhost:5000/api/v1",
                "type": "string"
            },
            {
                "key": "jwt_token",
                "value": "",
                "type": "string"
            }
        ],
        "item": []
    }
    
    # Convert paths to Postman requests
    for path, methods in spec.get('paths', {}).items():
        folder = {
            "name": path.replace('/api/v1', '').strip('/'),
            "item": []
        }
        
        for method, operation in methods.items():
            if method.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                request = {
                    "name": operation.get('summary', f"{method.upper()} {path}"),
                    "request": {
                        "method": method.upper(),
                        "header": [
                            {
                                "key": "Content-Type",
                                "value": "application/json",
                                "type": "text"
                            }
                        ],
                        "url": {
                            "raw": "{{base_url}}" + path,
                            "host": ["{{base_url}}"],
                            "path": path.strip('/').split('/')
                        }
                    }
                }
                
                # Add request body for POST/PUT requests
                if method.upper() in ['POST', 'PUT', 'PATCH']:
                    request["request"]["body"] = {
                        "mode": "raw",
                        "raw": "{}",
                        "options": {
                            "raw": {
                                "language": "json"
                            }
                        }
                    }
                
                folder["item"].append(request)
        
        if folder["item"]:
            collection["item"].append(folder)
    
    return collection


def main():
    """Main function to generate all documentation."""
    
    print("Generating API documentation...")
    
    try:
        # Generate OpenAPI spec
        spec = generate_openapi_spec()
        
        # Save documentation
        save_documentation(spec)
        
        # Generate Postman collection
        postman_collection = generate_postman_collection(spec)
        postman_path = 'docs/api/postman_collection.json'
        os.makedirs(os.path.dirname(postman_path), exist_ok=True)
        
        with open(postman_path, 'w', encoding='utf-8') as f:
            json.dump(postman_collection, f, indent=2, ensure_ascii=False)
        
        print(f"  Postman Collection: {postman_path}")
        
        print("\nDocumentation generated successfully!")
        print("\nTo view the documentation:")
        print("1. Open docs/api/index.html in your browser")
        print("2. Or visit http://localhost:5000/api/v1/docs/ when the API server is running")
        print("3. Import docs/api/postman_collection.json into Postman for testing")
        
    except Exception as e:
        print(f"Error generating documentation: {str(e)}")
        raise


if __name__ == '__main__':
    main()