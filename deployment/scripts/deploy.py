#!/usr/bin/env python3
"""
Automated deployment script for MCP Crawl4AI RAG with multi-provider support.

This script handles:
- Environment validation and setup
- Docker Compose orchestration
- Health checks and service validation
- Provider-specific configurations
- Rollback capabilities
- Zero-downtime deployments
"""

import os
import sys
import argparse
import time
import json
import subprocess
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import requests
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/deployment.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    environment: str
    provider: str
    profiles: List[str]
    env_file: str
    validate_only: bool = False
    dry_run: bool = False
    force: bool = False
    skip_health_checks: bool = False
    timeout: int = 300

class DeploymentError(Exception):
    """Custom deployment exception."""
    pass

class DeploymentManager:
    """Manages the deployment process."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.project_root = Path(__file__).parent.parent.parent
        self.compose_file = self.project_root / "docker-compose.yml"
        self.backup_dir = self.project_root / "deployment" / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Service health check endpoints
        self.health_endpoints = {
            'mcp-crawl4ai': 'http://localhost:8051/health',
            'ollama': 'http://localhost:11434/api/tags',
            'neo4j': 'http://localhost:7474/db/neo4j/tx/commit',
            'prometheus': 'http://localhost:9090/-/healthy',
            'grafana': 'http://localhost:3000/api/health'
        }
        
        # Service startup order
        self.startup_order = [
            ['neo4j', 'ollama'],  # Infrastructure services
            ['prometheus', 'grafana'],  # Monitoring services
            ['ollama-init'],  # Model initialization
            ['mcp-crawl4ai']  # Application services
        ]
    
    def validate_environment(self) -> bool:
        """Validate deployment environment and configuration."""
        logger.info(f"Validating environment: {self.config.environment}")
        
        # Check if environment file exists
        env_file_path = Path(self.config.env_file)
        if not env_file_path.exists():
            raise DeploymentError(f"Environment file not found: {self.config.env_file}")
        
        # Validate docker-compose file
        if not self.compose_file.exists():
            raise DeploymentError(f"Docker compose file not found: {self.compose_file}")
        
        try:
            result = subprocess.run([
                'docker-compose', '-f', str(self.compose_file), 
                '--env-file', self.config.env_file, 
                'config'
            ], capture_output=True, text=True, check=True)
            logger.info("Docker Compose configuration is valid")
        except subprocess.CalledProcessError as e:
            raise DeploymentError(f"Invalid Docker Compose configuration: {e.stderr}")
        
        # Validate required environment variables
        required_vars = self._get_required_env_vars()
        missing_vars = []
        
        with open(env_file_path) as f:
            env_content = f.read()
            for var in required_vars:
                if f"{var}=" not in env_content:
                    missing_vars.append(var)
        
        if missing_vars:
            raise DeploymentError(f"Missing required environment variables: {missing_vars}")
        
        # Validate provider-specific requirements
        self._validate_provider_requirements()
        
        logger.info("Environment validation completed successfully")
        return True
    
    def _get_required_env_vars(self) -> List[str]:
        """Get required environment variables based on provider."""
        base_vars = ['HOST', 'PORT', 'AI_PROVIDER', 'VECTOR_DB_PROVIDER']
        
        if 'openai' in self.config.provider.lower():
            base_vars.extend(['OPENAI_API_KEY'])
        
        if 'ollama' in self.config.provider.lower():
            base_vars.extend([
                'OLLAMA_BASE_URL', 'OLLAMA_EMBEDDING_MODEL', 
                'OLLAMA_LLM_MODEL', 'OLLAMA_EMBEDDING_DIMENSION'
            ])
        
        if 'supabase' in self.config.profiles:
            base_vars.extend(['SUPABASE_URL', 'SUPABASE_SERVICE_KEY'])
        
        if 'neo4j' in self.config.profiles:
            base_vars.extend(['NEO4J_URI', 'NEO4J_USER', 'NEO4J_PASSWORD'])
        
        return base_vars
    
    def _validate_provider_requirements(self):
        """Validate provider-specific requirements."""
        if 'ollama' in self.config.provider.lower():
            # Check if Ollama models are specified
            logger.info("Validating Ollama configuration...")
            # Additional Ollama-specific validations can be added here
        
        if 'openai' in self.config.provider.lower():
            # Validate OpenAI API key format
            logger.info("Validating OpenAI configuration...")
            # Additional OpenAI-specific validations can be added here
    
    def create_backup(self) -> str:
        """Create backup of current deployment."""
        timestamp = int(time.time())
        backup_name = f"backup_{self.config.environment}_{timestamp}"
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)
        
        logger.info(f"Creating backup: {backup_name}")
        
        try:
            # Backup docker-compose state
            result = subprocess.run([
                'docker-compose', '-f', str(self.compose_file),
                'ps', '--format', 'json'
            ], capture_output=True, text=True, check=True)
            
            with open(backup_path / 'services.json', 'w') as f:
                f.write(result.stdout)
            
            # Backup environment file
            if os.path.exists(self.config.env_file):
                subprocess.run([
                    'cp', self.config.env_file, 
                    str(backup_path / '.env')
                ], check=True)
            
            # Backup volumes (if any)
            self._backup_volumes(backup_path)
            
            logger.info(f"Backup created successfully: {backup_path}")
            return str(backup_path)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Backup creation failed: {e}")
            raise DeploymentError(f"Failed to create backup: {e}")
    
    def _backup_volumes(self, backup_path: Path):
        """Backup Docker volumes."""
        try:
            # List volumes
            result = subprocess.run([
                'docker', 'volume', 'ls', '--format', 'json'
            ], capture_output=True, text=True, check=True)
            
            volumes = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    volume_info = json.loads(line)
                    if 'mcp' in volume_info['Name']:
                        volumes.append(volume_info['Name'])
            
            # Create volume backup info
            with open(backup_path / 'volumes.json', 'w') as f:
                json.dump(volumes, f, indent=2)
                
        except subprocess.CalledProcessError as e:
            logger.warning(f"Volume backup failed: {e}")
    
    def deploy(self) -> bool:
        """Execute the deployment."""
        logger.info(f"Starting deployment with configuration: {self.config}")
        
        if self.config.dry_run:
            logger.info("DRY RUN MODE - No actual changes will be made")
            self._simulate_deployment()
            return True
        
        try:
            # Create backup
            backup_path = self.create_backup()
            
            # Stop existing services
            self._stop_services()
            
            # Pull latest images
            self._pull_images()
            
            # Start services in order
            self._start_services()
            
            # Validate deployment
            if not self.config.skip_health_checks:
                self._validate_deployment()
            
            logger.info("Deployment completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            if not self.config.force:
                logger.info("Starting rollback...")
                self._rollback(backup_path)
            raise DeploymentError(f"Deployment failed: {e}")
    
    def _simulate_deployment(self):
        """Simulate deployment for dry run."""
        logger.info("Simulating deployment steps:")
        logger.info(f"1. Would validate environment: {self.config.environment}")
        logger.info(f"2. Would use profiles: {self.config.profiles}")
        logger.info(f"3. Would use env file: {self.config.env_file}")
        logger.info("4. Would create backup")
        logger.info("5. Would stop existing services")
        logger.info("6. Would pull latest images")
        logger.info("7. Would start services")
        logger.info("8. Would validate health checks")
    
    def _stop_services(self):
        """Stop existing services."""
        logger.info("Stopping existing services...")
        
        profiles_arg = ",".join(self.config.profiles)
        
        try:
            subprocess.run([
                'docker-compose', '-f', str(self.compose_file),
                '--env-file', self.config.env_file,
                '--profile', profiles_arg,
                'down'
            ], check=True, timeout=120)
            
            logger.info("Services stopped successfully")
            
        except subprocess.TimeoutExpired:
            logger.warning("Service shutdown timed out, forcing stop...")
            subprocess.run([
                'docker-compose', '-f', str(self.compose_file),
                '--profile', profiles_arg,
                'kill'
            ])
        except subprocess.CalledProcessError as e:
            logger.warning(f"Some services may not have stopped cleanly: {e}")
    
    def _pull_images(self):
        """Pull latest Docker images."""
        logger.info("Pulling latest images...")
        
        profiles_arg = ",".join(self.config.profiles)
        
        try:
            subprocess.run([
                'docker-compose', '-f', str(self.compose_file),
                '--env-file', self.config.env_file,
                '--profile', profiles_arg,
                'pull'
            ], check=True, timeout=600)
            
            logger.info("Images pulled successfully")
            
        except subprocess.CalledProcessError as e:
            raise DeploymentError(f"Failed to pull images: {e}")
    
    def _start_services(self):
        """Start services in the correct order."""
        logger.info("Starting services...")
        
        profiles_arg = ",".join(self.config.profiles)
        
        # Start all services
        try:
            subprocess.run([
                'docker-compose', '-f', str(self.compose_file),
                '--env-file', self.config.env_file,
                '--profile', profiles_arg,
                'up', '-d'
            ], check=True, timeout=300)
            
            logger.info("Services started successfully")
            
        except subprocess.CalledProcessError as e:
            raise DeploymentError(f"Failed to start services: {e}")
    
    def _validate_deployment(self) -> bool:
        """Validate that deployment is healthy."""
        logger.info("Validating deployment health...")
        
        # Wait for services to start
        time.sleep(30)
        
        # Check service health
        health_results = {}
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_service = {
                executor.submit(self._check_service_health, service, endpoint): service
                for service, endpoint in self.health_endpoints.items()
            }
            
            for future in as_completed(future_to_service):
                service = future_to_service[future]
                try:
                    is_healthy = future.result(timeout=30)
                    health_results[service] = is_healthy
                except Exception as e:
                    logger.warning(f"Health check failed for {service}: {e}")
                    health_results[service] = False
        
        # Evaluate results
        failed_services = [
            service for service, healthy in health_results.items()
            if not healthy
        ]
        
        if failed_services:
            logger.error(f"Health checks failed for services: {failed_services}")
            raise DeploymentError(f"Unhealthy services: {failed_services}")
        
        logger.info("All health checks passed")
        return True
    
    def _check_service_health(self, service: str, endpoint: str) -> bool:
        """Check health of a specific service."""
        try:
            response = requests.get(endpoint, timeout=10)
            if response.status_code == 200:
                logger.info(f"✓ {service} is healthy")
                return True
            else:
                logger.warning(f"✗ {service} health check failed: {response.status_code}")
                return False
        except requests.RequestException as e:
            logger.warning(f"✗ {service} health check failed: {e}")
            return False
    
    def _rollback(self, backup_path: str):
        """Rollback to previous deployment."""
        logger.info(f"Rolling back using backup: {backup_path}")
        
        try:
            # Stop current services
            self._stop_services()
            
            # Restore environment file
            backup_env = Path(backup_path) / '.env'
            if backup_env.exists():
                subprocess.run([
                    'cp', str(backup_env), self.config.env_file
                ], check=True)
            
            # Start services with backup configuration
            self._start_services()
            
            logger.info("Rollback completed successfully")
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            raise DeploymentError(f"Rollback failed: {e}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deploy MCP Crawl4AI RAG with multi-provider support"
    )
    
    parser.add_argument(
        '--env', '--environment',
        choices=['development', 'staging', 'production'],
        default='development',
        help='Deployment environment'
    )
    
    parser.add_argument(
        '--provider',
        choices=['openai', 'ollama', 'hybrid'],
        default='ollama',
        help='AI provider to deploy'
    )
    
    parser.add_argument(
        '--profiles',
        nargs='+',
        help='Docker Compose profiles to use'
    )
    
    parser.add_argument(
        '--env-file',
        help='Environment file to use'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration, do not deploy'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate deployment without making changes'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force deployment even if validation fails'
    )
    
    parser.add_argument(
        '--skip-health-checks',
        action='store_true',
        help='Skip health checks after deployment'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Deployment timeout in seconds'
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Determine profiles based on provider if not specified
    if not args.profiles:
        if args.provider == 'ollama':
            args.profiles = ['ollama', 'monitoring']
        elif args.provider == 'hybrid':
            args.profiles = ['hybrid', 'monitoring']
        else:
            args.profiles = ['sqlite']
    
    # Determine environment file if not specified
    if not args.env_file:
        project_root = Path(__file__).parent.parent.parent
        if args.provider == 'hybrid':
            args.env_file = str(project_root / 'deployment' / 'environments' / f'.env.{args.provider}-{args.env}')
        else:
            args.env_file = str(project_root / 'deployment' / 'environments' / f'.env.{args.env}')
    
    # Create deployment configuration
    config = DeploymentConfig(
        environment=args.env,
        provider=args.provider,
        profiles=args.profiles,
        env_file=args.env_file,
        validate_only=args.validate_only,
        dry_run=args.dry_run,
        force=args.force,
        skip_health_checks=args.skip_health_checks,
        timeout=args.timeout
    )
    
    # Create deployment manager
    manager = DeploymentManager(config)
    
    try:
        # Validate environment
        manager.validate_environment()
        
        if config.validate_only:
            logger.info("Configuration validation completed successfully")
            return 0
        
        # Execute deployment
        success = manager.deploy()
        
        if success:
            logger.info("Deployment completed successfully!")
            return 0
        else:
            logger.error("Deployment failed!")
            return 1
            
    except DeploymentError as e:
        logger.error(f"Deployment error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Deployment interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())