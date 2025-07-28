#!/usr/bin/env python3
"""
Deployment Script for MCP Crawl4AI with Multi-Provider Support
==============================================================

This script automates the deployment process for different AI provider
and database combinations with proper environment configuration.

Usage:
    python deploy_provider.py --provider ollama --database sqlite --env development
    python deploy_provider.py --provider hybrid --database supabase --env production
    python deploy_provider.py --list-profiles
"""

import argparse
import subprocess
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import yaml

# Configuration
SUPPORTED_PROVIDERS = {
    'openai': {
        'description': 'OpenAI GPT models with OpenAI embeddings',
        'requires': ['OPENAI_API_KEY'],
        'compose_profile': 'supabase'
    },
    'ollama': {
        'description': 'Local Ollama models for both LLM and embeddings',
        'requires': ['OLLAMA_BASE_URL'],
        'compose_profile': 'ollama-sqlite'
    },
    'hybrid': {
        'description': 'OpenAI embeddings + Ollama LLM',
        'requires': ['OPENAI_API_KEY', 'OLLAMA_BASE_URL'],
        'compose_profile': 'hybrid'
    },
    'mixed': {
        'description': 'Alias for hybrid configuration',
        'requires': ['OPENAI_API_KEY', 'OLLAMA_BASE_URL'],
        'compose_profile': 'hybrid'
    }
}

SUPPORTED_DATABASES = {
    'sqlite': {
        'description': 'Local SQLite database for development',
        'compose_profile': 'sqlite'
    },
    'supabase': {
        'description': 'Supabase vector database',
        'requires': ['SUPABASE_URL', 'SUPABASE_SERVICE_KEY'],
        'compose_profile': 'supabase'
    },
    'neo4j': {
        'description': 'Neo4j vector database with knowledge graph',
        'compose_profile': 'neo4j'
    },
    'pinecone': {
        'description': 'Pinecone vector database',
        'requires': ['PINECONE_API_KEY'],
        'compose_profile': 'pinecone'
    },
    'weaviate': {
        'description': 'Weaviate vector database',
        'compose_profile': 'weaviate'
    }
}

SUPPORTED_ENVIRONMENTS = ['development', 'production', 'ollama', 'hybrid']


class DeploymentManager:
    """Manages deployment of MCP Crawl4AI with different provider configurations."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.deployment_dir = project_root / 'deployment'
        self.env_dir = self.deployment_dir / 'environments'
        
    def list_profiles(self) -> None:
        """List all available deployment profiles."""
        print("\\n=== Available AI Providers ===")
        for provider, config in SUPPORTED_PROVIDERS.items():
            print(f"  {provider:10} - {config['description']}")
            if 'requires' in config:
                print(f"             Requires: {', '.join(config['requires'])}")
        
        print("\\n=== Available Databases ===")
        for db, config in SUPPORTED_DATABASES.items():
            print(f"  {db:10} - {config['description']}")
            if 'requires' in config:
                print(f"             Requires: {', '.join(config['requires'])}")
        
        print("\\n=== Available Environments ===")
        for env in SUPPORTED_ENVIRONMENTS:
            env_file = self.env_dir / f'.env.{env}'
            status = "✓" if env_file.exists() else "✗"
            print(f"  {status} {env}")
    
    def validate_environment(self, provider: str, database: str, env: str) -> Dict[str, str]:
        """Validate and load environment configuration."""
        # Load base environment file
        env_file = self.env_dir / f'.env.{env}'
        if not env_file.exists():
            raise FileNotFoundError(f"Environment file not found: {env_file}")
        
        env_vars = {}
        
        # Parse environment file
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value
        
        # Validate provider requirements
        if provider in SUPPORTED_PROVIDERS:
            required_vars = SUPPORTED_PROVIDERS[provider].get('requires', [])
            for var in required_vars:
                if var not in os.environ and var not in env_vars:
                    print(f"Warning: Required environment variable {var} not set")
        
        # Validate database requirements
        if database in SUPPORTED_DATABASES:
            required_vars = SUPPORTED_DATABASES[database].get('requires', [])
            for var in required_vars:
                if var not in os.environ and var not in env_vars:
                    print(f"Warning: Required environment variable {var} not set")
        
        return env_vars
    
    def determine_compose_profiles(self, provider: str, database: str) -> List[str]:
        """Determine Docker Compose profiles based on provider and database."""
        profiles = []
        
        # Add provider-specific profiles
        if provider == 'ollama':
            if database == 'sqlite':
                profiles.append('ollama-sqlite')
            elif database == 'neo4j':
                profiles.append('ollama-full')
            else:
                profiles.extend(['ollama', database])
        elif provider == 'hybrid':
            profiles.append('hybrid')
            if database != 'supabase':  # hybrid defaults to supabase
                profiles.append(database)
        else:
            # OpenAI or other providers
            profiles.append(database)
        
        # Add monitoring if requested
        if os.getenv('ENABLE_MONITORING', '').lower() == 'true':
            profiles.append('monitoring')
        
        return profiles
    
    def deploy(self, provider: str, database: str, env: str, 
               detach: bool = True, build: bool = False, pull: bool = False) -> bool:
        """Deploy the application with specified configuration."""
        try:
            # Validate configuration
            print(f"Deploying MCP Crawl4AI with {provider} provider and {database} database...")
            env_vars = self.validate_environment(provider, database, env)
            
            # Determine compose profiles
            profiles = self.determine_compose_profiles(provider, database)
            print(f"Using Docker Compose profiles: {', '.join(profiles)}")
            
            # Build Docker Compose command
            compose_cmd = ['docker', 'compose']
            
            # Add environment file
            env_file = self.env_dir / f'.env.{env}'
            compose_cmd.extend(['--env-file', str(env_file)])
            
            # Set profiles
            if profiles:
                os.environ['COMPOSE_PROFILES'] = ','.join(profiles)
            
            # Add build flag if requested
            if build:
                print("Building images...")
                build_cmd = compose_cmd + ['build']
                subprocess.run(build_cmd, check=True, cwd=self.project_root)
            
            # Add pull flag if requested
            if pull:
                print("Pulling latest images...")
                pull_cmd = compose_cmd + ['pull']
                subprocess.run(pull_cmd, check=False, cwd=self.project_root)  # Non-critical
            
            # Deploy services
            deploy_cmd = compose_cmd + ['up']
            if detach:
                deploy_cmd.append('-d')
            
            print(f"Running: {' '.join(deploy_cmd)}")
            result = subprocess.run(deploy_cmd, cwd=self.project_root)
            
            if result.returncode == 0:
                print("\\n✓ Deployment successful!")
                self.show_deployment_status(profiles)
                return True
            else:
                print("\\n✗ Deployment failed!")
                return False
                
        except Exception as e:
            print(f"Deployment failed: {e}")
            return False
    
    def show_deployment_status(self, profiles: List[str]) -> None:
        """Show status of deployed services."""
        print("\\n=== Deployment Status ===")
        
        # Show running containers
        try:
            result = subprocess.run(
                ['docker', 'compose', 'ps', '--format', 'json'], 
                capture_output=True, text=True, cwd=self.project_root
            )
            
            if result.returncode == 0 and result.stdout.strip():
                services = json.loads(result.stdout)
                if isinstance(services, list):
                    for service in services:
                        name = service.get('Name', 'Unknown')
                        status = service.get('State', 'Unknown')
                        ports = service.get('Publishers', [])
                        port_info = ', '.join([f"{p.get('PublishedPort', '?')}:{p.get('TargetPort', '?')}" for p in ports]) if ports else 'No ports'
                        print(f"  {name:30} {status:10} {port_info}")
        except Exception as e:
            print(f"Could not get service status: {e}")
        
        # Show access URLs
        print("\\n=== Access Information ===")
        print("  MCP Server:           http://localhost:8051")
        if 'ollama' in profiles:
            print("  Ollama API:           http://localhost:11434")
        if 'neo4j' in profiles:
            print("  Neo4j Browser:        http://localhost:7474")
        if 'weaviate' in profiles:
            print("  Weaviate:             http://localhost:8080")
        if 'monitoring' in profiles:
            print("  Prometheus:           http://localhost:9090")
            print("  Grafana:              http://localhost:3000 (admin/admin)")
    
    def stop(self) -> bool:
        """Stop all running services."""
        try:
            result = subprocess.run(
                ['docker', 'compose', 'down'], 
                cwd=self.project_root
            )
            return result.returncode == 0
        except Exception as e:
            print(f"Failed to stop services: {e}")
            return False
    
    def logs(self, service: Optional[str] = None, follow: bool = False) -> None:
        """Show logs for services."""
        try:
            cmd = ['docker', 'compose', 'logs']
            if follow:
                cmd.append('-f')
            if service:
                cmd.append(service)
            
            subprocess.run(cmd, cwd=self.project_root)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Failed to show logs: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Deploy MCP Crawl4AI with multi-provider support',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--provider', '-p',
        choices=list(SUPPORTED_PROVIDERS.keys()),
        default='openai',
        help='AI provider to use'
    )
    
    parser.add_argument(
        '--database', '-d',
        choices=list(SUPPORTED_DATABASES.keys()),
        default='sqlite',
        help='Vector database to use'
    )
    
    parser.add_argument(
        '--env', '-e',
        choices=SUPPORTED_ENVIRONMENTS,
        default='development',
        help='Environment configuration'
    )
    
    parser.add_argument(
        '--build', '-b',
        action='store_true',
        help='Build images before deploying'
    )
    
    parser.add_argument(
        '--pull',
        action='store_true',
        help='Pull latest images before deploying'
    )
    
    parser.add_argument(
        '--foreground', '-f',
        action='store_true',
        help='Run in foreground (default: detached)'
    )
    
    parser.add_argument(
        '--list-profiles', '-l',
        action='store_true',
        help='List available profiles and exit'
    )
    
    parser.add_argument(
        '--stop',
        action='store_true',
        help='Stop all running services'
    )
    
    parser.add_argument(
        '--logs',
        metavar='SERVICE',
        nargs='?',
        const='',
        help='Show logs for services (optionally specify service name)'
    )
    
    parser.add_argument(
        '--follow',
        action='store_true',
        help='Follow logs (use with --logs)'
    )
    
    args = parser.parse_args()
    
    # Find project root
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent  # Go up from deployment/scripts/
    
    manager = DeploymentManager(project_root)
    
    if args.list_profiles:
        manager.list_profiles()
        return
    
    if args.stop:
        if manager.stop():
            print("✓ Services stopped successfully")
        else:
            print("✗ Failed to stop services")
            sys.exit(1)
        return
    
    if args.logs is not None:
        service = args.logs if args.logs else None
        manager.logs(service, args.follow)
        return
    
    # Deploy with specified configuration
    success = manager.deploy(
        provider=args.provider,
        database=args.database,
        env=args.env,
        detach=not args.foreground,
        build=args.build,
        pull=args.pull
    )
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()