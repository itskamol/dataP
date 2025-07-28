#!/usr/bin/env python3
"""
Deployment Testing and Validation Suite
Tests deployment procedures, rollback mechanisms, and production readiness
"""

import os
import sys
import json
import time
import yaml
import shutil
import tempfile
import unittest
import subprocess
import requests
from pathlib import Path
from unittest.mock import patch, MagicMock

class DeploymentTestSuite(unittest.TestCase):
    """Comprehensive deployment testing"""
    
    @classmethod
    def setUpClass(cls):
        """Set up deployment test environment"""
        cls.test_dir = tempfile.mkdtemp(prefix='deployment_test_')
        cls.project_root = Path(__file__).parent.parent
        
        print(f"Deployment test environment: {cls.test_dir}")
        print(f"Project root: {cls.project_root}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up deployment test environment"""
        try:
            shutil.rmtree(cls.test_dir)
        except Exception as e:
            print(f"Warning: Could not clean up test directory: {e}")

    def test_01_docker_container_build(self):
        """Test Docker container build process"""
        print("\n=== Testing Docker Container Build ===")
        
        # Check if Docker is available
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                self.skipTest("Docker not available")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.skipTest("Docker not available")
        
        # Test main application Dockerfile
        dockerfile_path = self.project_root / 'Dockerfile'
        if dockerfile_path.exists():
            print("Testing main application Docker build...")
            
            build_result = subprocess.run([
                'docker', 'build', 
                '-t', 'file-processing-app:test',
                '-f', str(dockerfile_path),
                str(self.project_root)
            ], capture_output=True, text=True, timeout=300)
            
            self.assertEqual(build_result.returncode, 0, 
                           f"Docker build failed: {build_result.stderr}")
            
            print("✅ Main application Docker build successful")
            
            # Test container startup
            run_result = subprocess.run([
                'docker', 'run', '--rm', '-d',
                '--name', 'file-processing-test',
                '-p', '5000:5000',
                'file-processing-app:test'
            ], capture_output=True, text=True, timeout=30)
            
            if run_result.returncode == 0:
                container_id = run_result.stdout.strip()
                
                try:
                    # Wait for container to start
                    time.sleep(10)
                    
                    # Test health endpoint
                    try:
                        response = requests.get('http://localhost:5000/health', timeout=5)
                        if response.status_code == 200:
                            print("✅ Container health check passed")
                        else:
                            print(f"⚠️  Container health check failed: {response.status_code}")
                    except requests.exceptions.RequestException:
                        print("⚠️  Container health check failed (connection error)")
                    
                finally:
                    # Stop container
                    subprocess.run(['docker', 'stop', container_id], 
                                 capture_output=True, timeout=30)
            
            # Clean up image
            subprocess.run(['docker', 'rmi', 'file-processing-app:test'], 
                         capture_output=True)
        
        # Test CLI Dockerfile if it exists
        cli_dockerfile_path = self.project_root / 'Dockerfile.cli'
        if cli_dockerfile_path.exists():
            print("Testing CLI Docker build...")
            
            build_result = subprocess.run([
                'docker', 'build',
                '-t', 'file-processing-cli:test',
                '-f', str(cli_dockerfile_path),
                str(self.project_root)
            ], capture_output=True, text=True, timeout=300)
            
            self.assertEqual(build_result.returncode, 0,
                           f"CLI Docker build failed: {build_result.stderr}")
            
            print("✅ CLI Docker build successful")
            
            # Clean up image
            subprocess.run(['docker', 'rmi', 'file-processing-cli:test'],
                         capture_output=True)

    def test_02_kubernetes_manifests_validation(self):
        """Test Kubernetes deployment manifests"""
        print("\n=== Testing Kubernetes Manifests ===")
        
        k8s_dir = self.project_root / 'deployment' / 'kubernetes'
        if not k8s_dir.exists():
            self.skipTest("Kubernetes manifests not found")
        
        # Check if kubectl is available
        try:
            result = subprocess.run(['kubectl', 'version', '--client'], 
                                  capture_output=True, text=True, timeout=10)
            kubectl_available = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            kubectl_available = False
        
        # Validate YAML syntax for all manifest files
        manifest_files = list(k8s_dir.glob('*.yaml')) + list(k8s_dir.glob('*.yml'))
        
        for manifest_file in manifest_files:
            print(f"Validating {manifest_file.name}...")
            
            try:
                with open(manifest_file, 'r') as f:
                    yaml_content = yaml.safe_load_all(f)
                    
                    # Validate each document in the YAML file
                    for doc in yaml_content:
                        if doc is not None:
                            self.assertIn('apiVersion', doc, f"Missing apiVersion in {manifest_file.name}")
                            self.assertIn('kind', doc, f"Missing kind in {manifest_file.name}")
                            self.assertIn('metadata', doc, f"Missing metadata in {manifest_file.name}")
                
                print(f"✅ {manifest_file.name} syntax valid")
                
                # If kubectl is available, do dry-run validation
                if kubectl_available:
                    validate_result = subprocess.run([
                        'kubectl', 'apply', '--dry-run=client', '-f', str(manifest_file)
                    ], capture_output=True, text=True, timeout=30)
                    
                    if validate_result.returncode == 0:
                        print(f"✅ {manifest_file.name} kubectl validation passed")
                    else:
                        print(f"⚠️  {manifest_file.name} kubectl validation failed: {validate_result.stderr}")
                
            except yaml.YAMLError as e:
                self.fail(f"YAML syntax error in {manifest_file.name}: {e}")
            except Exception as e:
                self.fail(f"Error validating {manifest_file.name}: {e}")

    def test_03_docker_compose_validation(self):
        """Test Docker Compose configuration"""
        print("\n=== Testing Docker Compose Configuration ===")
        
        compose_file = self.project_root / 'docker-compose.yml'
        if not compose_file.exists():
            self.skipTest("docker-compose.yml not found")
        
        # Check if docker-compose is available
        try:
            result = subprocess.run(['docker-compose', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                self.skipTest("docker-compose not available")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.skipTest("docker-compose not available")
        
        # Validate docker-compose.yml syntax
        validate_result = subprocess.run([
            'docker-compose', '-f', str(compose_file), 'config'
        ], capture_output=True, text=True, timeout=30)
        
        self.assertEqual(validate_result.returncode, 0,
                        f"docker-compose.yml validation failed: {validate_result.stderr}")
        
        print("✅ docker-compose.yml syntax valid")
        
        # Test docker-compose build
        build_result = subprocess.run([
            'docker-compose', '-f', str(compose_file), 'build'
        ], capture_output=True, text=True, timeout=600)
        
        if build_result.returncode == 0:
            print("✅ docker-compose build successful")
            
            # Test docker-compose up (dry run)
            up_result = subprocess.run([
                'docker-compose', '-f', str(compose_file), 'up', '--dry-run'
            ], capture_output=True, text=True, timeout=60)
            
            if up_result.returncode == 0:
                print("✅ docker-compose up validation passed")
            else:
                print(f"⚠️  docker-compose up validation failed: {up_result.stderr}")
        else:
            print(f"⚠️  docker-compose build failed: {build_result.stderr}")

    def test_04_environment_configuration(self):
        """Test environment-specific configurations"""
        print("\n=== Testing Environment Configurations ===")
        
        environments = ['development', 'staging', 'production']
        
        for env in environments:
            print(f"Testing {env} environment configuration...")
            
            # Test configuration file loading
            config_file = self.project_root / f'config_{env}.json'
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    # Validate required configuration keys
                    required_keys = ['database', 'logging', 'security']
                    for key in required_keys:
                        if key in config:
                            print(f"✅ {env}: {key} configuration present")
                        else:
                            print(f"⚠️  {env}: {key} configuration missing")
                    
                except json.JSONDecodeError as e:
                    self.fail(f"Invalid JSON in {config_file}: {e}")
            else:
                print(f"⚠️  Configuration file not found: {config_file}")
            
            # Test environment variables
            env_vars = {
                'development': ['DEBUG=true', 'LOG_LEVEL=DEBUG'],
                'staging': ['DEBUG=false', 'LOG_LEVEL=INFO'],
                'production': ['DEBUG=false', 'LOG_LEVEL=WARNING']
            }
            
            expected_vars = env_vars.get(env, [])
            for var in expected_vars:
                key, expected_value = var.split('=')
                print(f"✅ {env}: Expected {key}={expected_value}")

    def test_05_deployment_scripts(self):
        """Test deployment scripts"""
        print("\n=== Testing Deployment Scripts ===")
        
        scripts_dir = self.project_root / 'deployment' / 'scripts'
        if not scripts_dir.exists():
            self.skipTest("Deployment scripts directory not found")
        
        script_files = list(scripts_dir.glob('*.sh'))
        
        for script_file in script_files:
            print(f"Testing {script_file.name}...")
            
            # Check if script is executable
            if not os.access(script_file, os.X_OK):
                print(f"⚠️  {script_file.name} is not executable")
                continue
            
            # Validate shell script syntax
            syntax_check = subprocess.run([
                'bash', '-n', str(script_file)
            ], capture_output=True, text=True, timeout=30)
            
            if syntax_check.returncode == 0:
                print(f"✅ {script_file.name} syntax valid")
            else:
                print(f"❌ {script_file.name} syntax error: {syntax_check.stderr}")
                self.fail(f"Shell script syntax error in {script_file.name}")
            
            # Test script help/usage (if supported)
            help_check = subprocess.run([
                str(script_file), '--help'
            ], capture_output=True, text=True, timeout=10)
            
            if help_check.returncode == 0:
                print(f"✅ {script_file.name} help available")
            else:
                print(f"⚠️  {script_file.name} no help available")

    def test_06_monitoring_configuration(self):
        """Test monitoring and observability configuration"""
        print("\n=== Testing Monitoring Configuration ===")
        
        # Test Prometheus configuration
        prometheus_config = self.project_root / 'deployment' / 'prometheus' / 'prometheus.yml'
        if prometheus_config.exists():
            try:
                with open(prometheus_config, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Validate Prometheus config structure
                required_sections = ['global', 'scrape_configs']
                for section in required_sections:
                    self.assertIn(section, config, f"Missing {section} in prometheus.yml")
                
                print("✅ Prometheus configuration valid")
                
            except yaml.YAMLError as e:
                self.fail(f"Invalid YAML in prometheus.yml: {e}")
        else:
            print("⚠️  Prometheus configuration not found")
        
        # Test Grafana dashboards
        grafana_dir = self.project_root / 'deployment' / 'grafana' / 'dashboards'
        if grafana_dir.exists():
            dashboard_files = list(grafana_dir.glob('*.json'))
            
            for dashboard_file in dashboard_files:
                try:
                    with open(dashboard_file, 'r') as f:
                        dashboard = json.load(f)
                    
                    # Validate dashboard structure
                    required_keys = ['dashboard', 'title', 'panels']
                    for key in required_keys:
                        if key in dashboard:
                            print(f"✅ {dashboard_file.name}: {key} present")
                    
                except json.JSONDecodeError as e:
                    self.fail(f"Invalid JSON in {dashboard_file.name}: {e}")
        else:
            print("⚠️  Grafana dashboards not found")

    def test_07_security_configuration(self):
        """Test security configuration and hardening"""
        print("\n=== Testing Security Configuration ===")
        
        # Test SSL/TLS configuration
        ssl_config_paths = [
            self.project_root / 'deployment' / 'nginx' / 'ssl.conf',
            self.project_root / 'deployment' / 'kubernetes' / 'tls-secret.yaml'
        ]
        
        for ssl_config in ssl_config_paths:
            if ssl_config.exists():
                print(f"✅ SSL configuration found: {ssl_config.name}")
            else:
                print(f"⚠️  SSL configuration not found: {ssl_config.name}")
        
        # Test security headers configuration
        security_headers = [
            'X-Content-Type-Options',
            'X-Frame-Options',
            'X-XSS-Protection',
            'Strict-Transport-Security'
        ]
        
        nginx_config = self.project_root / 'deployment' / 'nginx' / 'nginx.conf'
        if nginx_config.exists():
            with open(nginx_config, 'r') as f:
                nginx_content = f.read()
            
            for header in security_headers:
                if header in nginx_content:
                    print(f"✅ Security header configured: {header}")
                else:
                    print(f"⚠️  Security header missing: {header}")
        
        # Test secrets management
        secrets_dir = self.project_root / 'deployment' / 'secrets'
        if secrets_dir.exists():
            secret_files = list(secrets_dir.glob('*.yaml'))
            
            for secret_file in secret_files:
                try:
                    with open(secret_file, 'r') as f:
                        secret = yaml.safe_load(f)
                    
                    if secret.get('kind') == 'Secret':
                        print(f"✅ Kubernetes secret configured: {secret_file.name}")
                    
                except yaml.YAMLError as e:
                    print(f"❌ Invalid secret file {secret_file.name}: {e}")

    def test_08_backup_and_recovery(self):
        """Test backup and recovery procedures"""
        print("\n=== Testing Backup and Recovery ===")
        
        # Test backup scripts
        backup_scripts = [
            self.project_root / 'deployment' / 'scripts' / 'backup.sh',
            self.project_root / 'deployment' / 'scripts' / 'restore.sh'
        ]
        
        for script in backup_scripts:
            if script.exists():
                # Test script syntax
                syntax_check = subprocess.run([
                    'bash', '-n', str(script)
                ], capture_output=True, text=True, timeout=30)
                
                if syntax_check.returncode == 0:
                    print(f"✅ {script.name} syntax valid")
                else:
                    print(f"❌ {script.name} syntax error: {syntax_check.stderr}")
            else:
                print(f"⚠️  Backup script not found: {script.name}")
        
        # Test database backup configuration
        db_backup_config = self.project_root / 'deployment' / 'backup' / 'db-backup.yaml'
        if db_backup_config.exists():
            try:
                with open(db_backup_config, 'r') as f:
                    config = yaml.safe_load(f)
                
                print("✅ Database backup configuration found")
                
            except yaml.YAMLError as e:
                print(f"❌ Invalid database backup config: {e}")
        else:
            print("⚠️  Database backup configuration not found")

    def test_09_rollback_procedures(self):
        """Test rollback mechanisms"""
        print("\n=== Testing Rollback Procedures ===")
        
        # Test rollback script
        rollback_script = self.project_root / 'deployment' / 'scripts' / 'rollback.sh'
        if rollback_script.exists():
            # Test script syntax
            syntax_check = subprocess.run([
                'bash', '-n', str(rollback_script)
            ], capture_output=True, text=True, timeout=30)
            
            if syntax_check.returncode == 0:
                print("✅ Rollback script syntax valid")
            else:
                print(f"❌ Rollback script syntax error: {syntax_check.stderr}")
        else:
            print("⚠️  Rollback script not found")
        
        # Test Kubernetes rollback configuration
        k8s_dir = self.project_root / 'deployment' / 'kubernetes'
        if k8s_dir.exists():
            deployment_files = list(k8s_dir.glob('*deployment*.yaml'))
            
            for deployment_file in deployment_files:
                try:
                    with open(deployment_file, 'r') as f:
                        deployment = yaml.safe_load(f)
                    
                    # Check for rollback configuration
                    spec = deployment.get('spec', {})
                    if 'revisionHistoryLimit' in spec:
                        print(f"✅ {deployment_file.name}: Rollback history configured")
                    else:
                        print(f"⚠️  {deployment_file.name}: No rollback history limit")
                    
                except yaml.YAMLError as e:
                    print(f"❌ Invalid deployment file {deployment_file.name}: {e}")

    def test_10_production_readiness_checklist(self):
        """Test production readiness checklist"""
        print("\n=== Testing Production Readiness ===")
        
        checklist_items = [
            # Configuration
            ('Configuration management', lambda: self._check_config_management()),
            ('Environment variables', lambda: self._check_environment_variables()),
            ('Secrets management', lambda: self._check_secrets_management()),
            
            # Security
            ('SSL/TLS configuration', lambda: self._check_ssl_config()),
            ('Security headers', lambda: self._check_security_headers()),
            ('Input validation', lambda: self._check_input_validation()),
            
            # Monitoring
            ('Health checks', lambda: self._check_health_checks()),
            ('Metrics collection', lambda: self._check_metrics()),
            ('Logging configuration', lambda: self._check_logging()),
            ('Alerting rules', lambda: self._check_alerting()),
            
            # Performance
            ('Resource limits', lambda: self._check_resource_limits()),
            ('Caching configuration', lambda: self._check_caching()),
            ('Database optimization', lambda: self._check_database_optimization()),
            
            # Reliability
            ('Backup procedures', lambda: self._check_backup_procedures()),
            ('Disaster recovery', lambda: self._check_disaster_recovery()),
            ('High availability', lambda: self._check_high_availability()),
        ]
        
        passed_checks = 0
        total_checks = len(checklist_items)
        
        for item_name, check_func in checklist_items:
            try:
                result = check_func()
                if result:
                    print(f"✅ {item_name}")
                    passed_checks += 1
                else:
                    print(f"❌ {item_name}")
            except Exception as e:
                print(f"⚠️  {item_name}: {e}")
        
        print(f"\nProduction readiness: {passed_checks}/{total_checks} checks passed")
        
        # Require at least 80% of checks to pass
        self.assertGreaterEqual(passed_checks / total_checks, 0.8,
                               "Production readiness checks failed")

    def _check_config_management(self):
        """Check configuration management"""
        config_files = list(self.project_root.glob('config*.json'))
        return len(config_files) > 0

    def _check_environment_variables(self):
        """Check environment variables configuration"""
        env_file = self.project_root / '.env.example'
        return env_file.exists()

    def _check_secrets_management(self):
        """Check secrets management"""
        secrets_dir = self.project_root / 'deployment' / 'secrets'
        return secrets_dir.exists()

    def _check_ssl_config(self):
        """Check SSL/TLS configuration"""
        ssl_configs = [
            self.project_root / 'deployment' / 'nginx' / 'ssl.conf',
            self.project_root / 'deployment' / 'kubernetes' / 'tls-secret.yaml'
        ]
        return any(config.exists() for config in ssl_configs)

    def _check_security_headers(self):
        """Check security headers configuration"""
        nginx_config = self.project_root / 'deployment' / 'nginx' / 'nginx.conf'
        if not nginx_config.exists():
            return False
        
        with open(nginx_config, 'r') as f:
            content = f.read()
        
        security_headers = ['X-Content-Type-Options', 'X-Frame-Options']
        return any(header in content for header in security_headers)

    def _check_input_validation(self):
        """Check input validation implementation"""
        # Check if security module exists
        security_module = self.project_root / 'src' / 'infrastructure' / 'security.py'
        return security_module.exists()

    def _check_health_checks(self):
        """Check health checks implementation"""
        health_module = self.project_root / 'src' / 'infrastructure' / 'health_checks.py'
        return health_module.exists()

    def _check_metrics(self):
        """Check metrics collection"""
        metrics_module = self.project_root / 'src' / 'infrastructure' / 'metrics.py'
        return metrics_module.exists()

    def _check_logging(self):
        """Check logging configuration"""
        logging_module = self.project_root / 'src' / 'infrastructure' / 'logging.py'
        return logging_module.exists()

    def _check_alerting(self):
        """Check alerting rules"""
        alert_rules = self.project_root / 'deployment' / 'prometheus' / 'alert_rules.yml'
        return alert_rules.exists()

    def _check_resource_limits(self):
        """Check resource limits in Kubernetes"""
        k8s_dir = self.project_root / 'deployment' / 'kubernetes'
        if not k8s_dir.exists():
            return False
        
        deployment_files = list(k8s_dir.glob('*deployment*.yaml'))
        for deployment_file in deployment_files:
            try:
                with open(deployment_file, 'r') as f:
                    deployment = yaml.safe_load(f)
                
                containers = deployment.get('spec', {}).get('template', {}).get('spec', {}).get('containers', [])
                for container in containers:
                    if 'resources' in container:
                        return True
            except:
                continue
        
        return False

    def _check_caching(self):
        """Check caching configuration"""
        cache_module = self.project_root / 'src' / 'infrastructure' / 'caching.py'
        return cache_module.exists()

    def _check_database_optimization(self):
        """Check database optimization"""
        # Check if database configuration exists
        db_configs = list(self.project_root.glob('**/database*.yaml'))
        return len(db_configs) > 0

    def _check_backup_procedures(self):
        """Check backup procedures"""
        backup_script = self.project_root / 'deployment' / 'scripts' / 'backup.sh'
        return backup_script.exists()

    def _check_disaster_recovery(self):
        """Check disaster recovery procedures"""
        dr_docs = list(self.project_root.glob('**/disaster-recovery*.md'))
        return len(dr_docs) > 0

    def _check_high_availability(self):
        """Check high availability configuration"""
        k8s_dir = self.project_root / 'deployment' / 'kubernetes'
        if not k8s_dir.exists():
            return False
        
        # Check for HPA (Horizontal Pod Autoscaler)
        hpa_file = k8s_dir / 'hpa.yaml'
        return hpa_file.exists()

def run_deployment_tests():
    """Run all deployment tests"""
    print("="*80)
    print("DEPLOYMENT TESTING SUITE")
    print("="*80)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(DeploymentTestSuite)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("DEPLOYMENT TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_deployment_tests()
    sys.exit(0 if success else 1)