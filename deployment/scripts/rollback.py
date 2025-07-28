#!/usr/bin/env python3
"""
Rollback and failover management script for MCP Crawl4AI RAG deployment.

This script handles:
- Automatic rollback to previous working deployment
- Provider failover (OpenAI <-> Ollama)
- Configuration rollback
- Database state management
- Service health monitoring during rollback
- Emergency procedures
"""

import os
import sys
import time
import json
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import requests
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/rollback.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class RollbackState:
    """Represents the state of a rollback operation."""
    rollback_id: str
    timestamp: float
    environment: str
    target_version: str
    current_version: str
    rollback_type: str  # 'deployment', 'provider', 'configuration'
    status: str  # 'initiated', 'in_progress', 'completed', 'failed'
    backup_path: str
    services_affected: List[str]
    health_checks_passed: bool = False
    error_message: str = None

class RollbackManager:
    """Manages rollback and failover operations."""
    
    def __init__(self, environment: str):
        self.environment = environment
        self.project_root = Path(__file__).parent.parent.parent
        self.compose_file = self.project_root / "docker-compose.yml"
        self.backup_dir = self.project_root / "deployment" / "backups"
        self.state_file = self.backup_dir / "rollback_state.json"
        self.deployment_history_file = self.backup_dir / "deployment_history.json"
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Service configuration
        self.service_configs = {
            'openai': {
                'profiles': ['sqlite'],
                'env_template': '.env.development',
                'services': ['mcp-crawl4ai-sqlite']
            },
            'ollama': {
                'profiles': ['ollama', 'monitoring'],
                'env_template': '.env.ollama-production',
                'services': ['ollama', 'ollama-init', 'mcp-crawl4ai-ollama-sqlite']
            },
            'hybrid': {
                'profiles': ['hybrid', 'monitoring'],
                'env_template': '.env.hybrid-production',
                'services': ['ollama', 'mcp-crawl4ai-hybrid']
            }
        }
        
        # Health check endpoints
        self.health_endpoints = {
            'mcp-crawl4ai': 'http://localhost:8051/health',
            'ollama': 'http://localhost:11434/api/tags',
            'prometheus': 'http://localhost:9090/-/healthy'
        }
    
    def get_deployment_history(self) -> List[Dict]:
        """Get deployment history."""
        if not self.deployment_history_file.exists():
            return []
        
        try:
            with open(self.deployment_history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load deployment history: {e}")
            return []
    
    def save_deployment_history(self, history: List[Dict]):
        """Save deployment history."""
        try:
            with open(self.deployment_history_file, 'w') as f:
                json.dump(history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save deployment history: {e}")
    
    def record_deployment(self, provider: str, version: str, config_path: str):
        """Record a successful deployment."""
        history = self.get_deployment_history()
        
        deployment_record = {
            'timestamp': time.time(),
            'environment': self.environment,
            'provider': provider,
            'version': version,
            'config_path': config_path,
            'services': self.service_configs.get(provider, {}).get('services', []),
            'health_validated': True
        }
        
        history.append(deployment_record)
        
        # Keep only last 10 deployments
        if len(history) > 10:
            history = history[-10:]
        
        self.save_deployment_history(history)
        logger.info(f"Recorded deployment: {provider} v{version}")
    
    def get_rollback_target(self, steps_back: int = 1) -> Optional[Dict]:
        """Get the target deployment for rollback."""
        history = self.get_deployment_history()
        
        if len(history) < steps_back + 1:
            logger.error(f"Not enough deployment history for rollback ({len(history)} deployments)")
            return None
        
        # Get the deployment that is 'steps_back' deployments ago
        target = history[-(steps_back + 1)]
        logger.info(f"Rollback target: {target['provider']} v{target['version']} from {datetime.fromtimestamp(target['timestamp'])}")
        
        return target
    
    def save_rollback_state(self, state: RollbackState):
        """Save rollback state."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(asdict(state), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save rollback state: {e}")
    
    def load_rollback_state(self) -> Optional[RollbackState]:
        """Load rollback state."""
        if not self.state_file.exists():
            return None
        
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
                return RollbackState(**data)
        except Exception as e:
            logger.error(f"Failed to load rollback state: {e}")
            return None
    
    def emergency_stop(self) -> bool:
        """Emergency stop of all services."""
        logger.warning("Initiating emergency stop of all services...")
        
        try:
            # Stop all Docker Compose services
            subprocess.run([
                'docker-compose', '-f', str(self.compose_file),
                'down', '--remove-orphans'
            ], check=True, timeout=60)
            
            logger.info("Emergency stop completed")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Emergency stop timed out, forcing kill...")
            try:
                subprocess.run([
                    'docker-compose', '-f', str(self.compose_file),
                    'kill'
                ], timeout=30)
                return True
            except Exception as e:
                logger.error(f"Force kill failed: {e}")
                return False
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            return False
    
    def rollback_deployment(self, target: Dict, rollback_type: str = 'deployment') -> RollbackState:
        """Perform deployment rollback."""
        rollback_id = f"rollback_{int(time.time())}"
        logger.info(f"Starting rollback operation: {rollback_id}")
        
        state = RollbackState(
            rollback_id=rollback_id,
            timestamp=time.time(),
            environment=self.environment,
            target_version=target['version'],
            current_version='unknown',
            rollback_type=rollback_type,
            status='initiated',
            backup_path='',
            services_affected=target['services']
        )
        
        try:
            # Update state
            state.status = 'in_progress'
            self.save_rollback_state(state)
            
            # Step 1: Create backup of current state
            logger.info("Creating backup of current state...")
            backup_path = self.create_current_backup()
            state.backup_path = backup_path
            
            # Step 2: Stop current services
            logger.info("Stopping current services...")
            self.stop_services()
            
            # Step 3: Restore target configuration
            logger.info("Restoring target configuration...")
            self.restore_configuration(target)
            
            # Step 4: Start services with target configuration
            logger.info("Starting services with rollback configuration...")
            self.start_services(target['provider'])
            
            # Step 5: Validate rollback
            logger.info("Validating rollback...")
            health_check_passed = self.validate_rollback(target)
            state.health_checks_passed = health_check_passed
            
            if health_check_passed:
                state.status = 'completed'
                logger.info(f"Rollback completed successfully: {rollback_id}")
            else:
                raise Exception("Health checks failed after rollback")
            
        except Exception as e:
            state.status = 'failed'
            state.error_message = str(e)
            logger.error(f"Rollback failed: {e}")
            
            # Attempt emergency recovery
            logger.info("Attempting emergency recovery...")
            self.emergency_recovery(state)
        
        finally:
            self.save_rollback_state(state)
        
        return state
    
    def provider_failover(self, from_provider: str, to_provider: str) -> RollbackState:
        """Perform provider failover."""
        rollback_id = f"failover_{from_provider}_to_{to_provider}_{int(time.time())}"
        logger.info(f"Starting provider failover: {rollback_id}")
        
        state = RollbackState(
            rollback_id=rollback_id,
            timestamp=time.time(),
            environment=self.environment,
            target_version=to_provider,
            current_version=from_provider,
            rollback_type='provider_failover',
            status='initiated',
            backup_path='',
            services_affected=[]
        )
        
        try:
            state.status = 'in_progress'
            self.save_rollback_state(state)
            
            # Validate target provider configuration
            if to_provider not in self.service_configs:
                raise Exception(f"Unsupported failover target: {to_provider}")
            
            target_config = self.service_configs[to_provider]
            state.services_affected = target_config['services']
            
            # Create backup
            backup_path = self.create_current_backup()
            state.backup_path = backup_path
            
            # Stop current services
            logger.info(f"Stopping {from_provider} services...")
            self.stop_services()
            
            # Switch configuration
            logger.info(f"Switching configuration to {to_provider}...")
            self.switch_provider_config(to_provider)
            
            # Start target services
            logger.info(f"Starting {to_provider} services...")
            self.start_services(to_provider)
            
            # Validate failover
            logger.info("Validating failover...")
            mock_target = {
                'provider': to_provider,
                'services': target_config['services']
            }
            health_check_passed = self.validate_rollback(mock_target)
            state.health_checks_passed = health_check_passed
            
            if health_check_passed:
                state.status = 'completed'
                logger.info(f"Provider failover completed: {from_provider} -> {to_provider}")
            else:
                raise Exception("Health checks failed after failover")
                
        except Exception as e:
            state.status = 'failed'
            state.error_message = str(e)
            logger.error(f"Provider failover failed: {e}")
            
            # Rollback to original provider
            logger.info(f"Rolling back to original provider: {from_provider}")
            try:
                self.stop_services()
                self.switch_provider_config(from_provider)
                self.start_services(from_provider)
            except Exception as rollback_error:
                logger.error(f"Rollback to original provider failed: {rollback_error}")
                self.emergency_stop()
        
        finally:
            self.save_rollback_state(state)
        
        return state
    
    def create_current_backup(self) -> str:
        """Create backup of current deployment state."""
        timestamp = int(time.time())
        backup_name = f"rollback_backup_{timestamp}"
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)
        
        try:
            # Backup Docker Compose state
            result = subprocess.run([
                'docker-compose', '-f', str(self.compose_file),
                'ps', '--format', 'json'
            ], capture_output=True, text=True, check=True)
            
            with open(backup_path / 'services.json', 'w') as f:
                f.write(result.stdout)
            
            # Backup current environment files
            env_dir = self.project_root / 'deployment' / 'environments'
            if env_dir.exists():
                shutil.copytree(env_dir, backup_path / 'environments')
            
            # Backup current configuration
            current_config = {
                'timestamp': timestamp,
                'environment': self.environment,
                'docker_compose_file': str(self.compose_file)
            }
            
            with open(backup_path / 'config.json', 'w') as f:
                json.dump(current_config, f, indent=2)
            
            logger.info(f"Backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            raise
    
    def restore_configuration(self, target: Dict):
        """Restore configuration from target deployment."""
        config_path = target.get('config_path')
        if config_path and Path(config_path).exists():
            # Copy target configuration
            current_env = self.project_root / '.env'
            shutil.copy2(config_path, current_env)
            logger.info(f"Restored configuration from: {config_path}")
        else:
            # Use template configuration
            provider = target['provider']
            template_name = self.service_configs[provider]['env_template']
            template_path = self.project_root / 'deployment' / 'environments' / template_name
            
            if template_path.exists():
                current_env = self.project_root / '.env'
                shutil.copy2(template_path, current_env)
                logger.info(f"Used template configuration: {template_path}")
            else:
                logger.warning("No configuration template found, using defaults")
    
    def switch_provider_config(self, provider: str):
        """Switch to provider-specific configuration."""
        template_name = self.service_configs[provider]['env_template']
        template_path = self.project_root / 'deployment' / 'environments' / template_name
        
        if template_path.exists():
            current_env = self.project_root / '.env'
            shutil.copy2(template_path, current_env)
            logger.info(f"Switched to {provider} configuration")
        else:
            raise Exception(f"Configuration template not found for {provider}: {template_path}")
    
    def stop_services(self):
        """Stop all services."""
        try:
            subprocess.run([
                'docker-compose', '-f', str(self.compose_file),
                'down', '--remove-orphans'
            ], check=True, timeout=120)
            
            logger.info("Services stopped successfully")
            
        except subprocess.TimeoutExpired:
            logger.warning("Service shutdown timed out, forcing stop...")
            subprocess.run([
                'docker-compose', '-f', str(self.compose_file),
                'kill'
            ])
        except Exception as e:
            logger.error(f"Failed to stop services: {e}")
            raise
    
    def start_services(self, provider: str):
        """Start services for the specified provider."""
        config = self.service_configs.get(provider, {})
        profiles = config.get('profiles', ['sqlite'])
        
        profiles_arg = ",".join(profiles)
        
        try:
            subprocess.run([
                'docker-compose', '-f', str(self.compose_file),
                '--profile', profiles_arg,
                'up', '-d'
            ], check=True, timeout=300)
            
            logger.info(f"Services started with profiles: {profiles}")
            
        except Exception as e:
            logger.error(f"Failed to start services: {e}")
            raise
    
    def validate_rollback(self, target: Dict) -> bool:
        """Validate that rollback was successful."""
        logger.info("Validating rollback health...")
        
        # Wait for services to start
        time.sleep(30)
        
        # Check service health
        all_healthy = True
        
        for service, endpoint in self.health_endpoints.items():
            try:
                response = requests.get(endpoint, timeout=10)
                if response.status_code == 200:
                    logger.info(f"✓ {service} is healthy")
                else:
                    logger.error(f"✗ {service} health check failed: {response.status_code}")
                    all_healthy = False
            except Exception as e:
                logger.error(f"✗ {service} health check failed: {e}")
                all_healthy = False
        
        # Check that expected services are running
        expected_services = target.get('services', [])
        for service in expected_services:
            try:
                result = subprocess.run([
                    'docker', 'ps', '--filter', f'name={service}', '--format', 'table {{.Names}}\t{{.Status}}'
                ], capture_output=True, text=True, check=True)
                
                if service in result.stdout and 'Up' in result.stdout:
                    logger.info(f"✓ {service} container is running")
                else:
                    logger.error(f"✗ {service} container is not running")
                    all_healthy = False
            except Exception as e:
                logger.error(f"✗ Failed to check {service} status: {e}")
                all_healthy = False
        
        return all_healthy
    
    def emergency_recovery(self, failed_state: RollbackState):
        """Perform emergency recovery after failed rollback."""
        logger.warning("Performing emergency recovery...")
        
        try:
            # Stop everything
            self.emergency_stop()
            
            # Try to restore from backup if available
            if failed_state.backup_path and Path(failed_state.backup_path).exists():
                logger.info("Attempting recovery from backup...")
                # This would be a more complex restoration process
                # For now, just restart with minimal configuration
                
            # Start with minimal safe configuration (OpenAI + SQLite)
            logger.info("Starting with minimal safe configuration...")
            self.switch_provider_config('openai')
            self.start_services('openai')
            
            # Validate minimal recovery
            time.sleep(30)
            response = requests.get('http://localhost:8051/health', timeout=10)
            if response.status_code == 200:
                logger.info("Emergency recovery successful")
            else:
                logger.error("Emergency recovery failed - manual intervention required")
                
        except Exception as e:
            logger.error(f"Emergency recovery failed: {e}")
            logger.error("CRITICAL: Manual intervention required!")
    
    def status(self) -> Dict:
        """Get current rollback/deployment status."""
        state = self.load_rollback_state()
        history = self.get_deployment_history()
        
        current_deployment = history[-1] if history else None
        
        return {
            'current_deployment': current_deployment,
            'rollback_state': asdict(state) if state else None,
            'deployment_history_count': len(history),
            'emergency_procedures_available': True
        }

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Rollback and failover management for MCP Crawl4AI RAG"
    )
    
    parser.add_argument(
        '--env', '--environment',
        choices=['development', 'staging', 'production'],
        default='development',
        help='Environment to operate on'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback deployment')
    rollback_parser.add_argument(
        '--steps', type=int, default=1,
        help='Number of deployments to rollback'
    )
    
    # Provider failover command
    failover_parser = subparsers.add_parser('failover', help='Provider failover')
    failover_parser.add_argument(
        '--from', dest='from_provider', required=True,
        choices=['openai', 'ollama', 'hybrid'],
        help='Current provider'
    )
    failover_parser.add_argument(
        '--to', dest='to_provider', required=True,
        choices=['openai', 'ollama', 'hybrid'],
        help='Target provider'
    )
    
    # Emergency commands
    emergency_parser = subparsers.add_parser('emergency', help='Emergency procedures')
    emergency_parser.add_argument(
        'action', choices=['stop', 'recover'],
        help='Emergency action to perform'
    )
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show rollback status')
    
    # Record deployment command
    record_parser = subparsers.add_parser('record', help='Record successful deployment')
    record_parser.add_argument('--provider', required=True, help='Provider name')
    record_parser.add_argument('--version', required=True, help='Version')
    record_parser.add_argument('--config', required=True, help='Config file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Create rollback manager
    manager = RollbackManager(args.env)
    
    try:
        if args.command == 'rollback':
            target = manager.get_rollback_target(args.steps)
            if not target:
                logger.error("No rollback target available")
                return 1
            
            state = manager.rollback_deployment(target)
            
            if state.status == 'completed':
                logger.info("Rollback completed successfully!")
                return 0
            else:
                logger.error(f"Rollback failed: {state.error_message}")
                return 1
        
        elif args.command == 'failover':
            state = manager.provider_failover(args.from_provider, args.to_provider)
            
            if state.status == 'completed':
                logger.info("Provider failover completed successfully!")
                return 0
            else:
                logger.error(f"Provider failover failed: {state.error_message}")
                return 1
        
        elif args.command == 'emergency':
            if args.action == 'stop':
                success = manager.emergency_stop()
                return 0 if success else 1
            elif args.action == 'recover':
                # Create a mock failed state for recovery
                failed_state = RollbackState(
                    rollback_id='emergency_recovery',
                    timestamp=time.time(),
                    environment=args.env,
                    target_version='safe',
                    current_version='unknown',
                    rollback_type='emergency',
                    status='failed',
                    backup_path='',
                    services_affected=[]
                )
                manager.emergency_recovery(failed_state)
                return 0
        
        elif args.command == 'status':
            status = manager.status()
            print(json.dumps(status, indent=2, default=str))
            return 0
        
        elif args.command == 'record':
            manager.record_deployment(args.provider, args.version, args.config)
            return 0
        
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())