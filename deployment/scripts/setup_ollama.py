#!/usr/bin/env python3
"""
Ollama Setup and Management Script for MCP Crawl4AI
===================================================

This script helps with:
- Installing and configuring Ollama
- Pulling required models
- Validating model availability
- Managing model updates
- Performance optimization

Usage:
    python setup_ollama.py --install
    python setup_ollama.py --pull-models
    python setup_ollama.py --validate
    python setup_ollama.py --optimize
"""

import argparse
import subprocess
import sys
import os
import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configurations
REQUIRED_MODELS = {
    'embedding': {
        'name': 'nomic-embed-text',
        'description': 'High-quality embedding model for text vectorization',
        'size': '274MB',
        'dimension': 768,
        'recommended': True
    },
    'llm': {
        'name': 'llama3.2:1b',
        'description': 'Compact LLM for text generation and analysis',
        'size': '1.3GB',
        'recommended': True
    },
    'llm_alternative': {
        'name': 'llama3.2:3b',
        'description': 'Larger LLM with better performance',
        'size': '2.0GB',
        'recommended': False
    },
    'code_llm': {
        'name': 'codellama:7b-code',
        'description': 'Specialized model for code understanding',
        'size': '3.8GB',
        'recommended': False
    }
}

# Performance configurations
PERFORMANCE_CONFIGS = {
    'low_memory': {
        'OLLAMA_NUM_PARALLEL': 1,
        'OLLAMA_MAX_LOADED_MODELS': 1,
        'OLLAMA_FLASH_ATTENTION': False,
        'description': 'Optimized for systems with limited RAM (< 8GB)'
    },
    'balanced': {
        'OLLAMA_NUM_PARALLEL': 2,
        'OLLAMA_MAX_LOADED_MODELS': 2,
        'OLLAMA_FLASH_ATTENTION': True,
        'description': 'Balanced performance and memory usage (8-16GB RAM)'
    },
    'high_performance': {
        'OLLAMA_NUM_PARALLEL': 4,
        'OLLAMA_MAX_LOADED_MODELS': 3,
        'OLLAMA_FLASH_ATTENTION': True,
        'description': 'Maximum performance for high-end systems (>16GB RAM)'
    }
}


class OllamaManager:
    """Manages Ollama installation, configuration, and models."""
    
    def __init__(self, base_url: str = 'http://localhost:11434'):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        
    def is_ollama_installed(self) -> bool:
        """Check if Ollama is installed."""
        try:
            result = subprocess.run(['ollama', '--version'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def is_ollama_running(self) -> bool:
        """Check if Ollama service is running."""
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def install_ollama(self) -> bool:
        """Install Ollama based on the operating system."""
        system = platform.system().lower()
        
        if self.is_ollama_installed():
            logger.info("Ollama is already installed")
            return True
        
        logger.info(f"Installing Ollama for {system}...")
        
        try:
            if system == 'darwin':  # macOS
                subprocess.run(['brew', 'install', 'ollama'], check=True)
            elif system == 'linux':
                # Download and run the install script
                install_cmd = 'curl -fsSL https://ollama.ai/install.sh | sh'
                subprocess.run(install_cmd, shell=True, check=True)
            else:
                logger.error(f"Automatic installation not supported for {system}")
                logger.info("Please install Ollama manually from https://ollama.ai/")
                return False
            
            logger.info("✓ Ollama installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install Ollama: {e}")
            return False
    
    def start_ollama_service(self) -> bool:
        """Start Ollama service."""
        if self.is_ollama_running():
            logger.info("Ollama service is already running")
            return True
        
        logger.info("Starting Ollama service...")
        
        try:
            # Try to start as a background service
            if platform.system().lower() == 'darwin':
                subprocess.Popen(['ollama', 'serve'], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
            else:
                # On Linux, might be managed by systemd
                try:
                    subprocess.run(['systemctl', 'start', 'ollama'], check=True)
                except subprocess.CalledProcessError:
                    # Fallback to direct execution
                    subprocess.Popen(['ollama', 'serve'],
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
            
            # Wait for service to start
            for i in range(30):
                if self.is_ollama_running():
                    logger.info("✓ Ollama service started successfully")
                    return True
                time.sleep(1)
            
            logger.error("Ollama service failed to start within 30 seconds")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start Ollama service: {e}")
            return False
    
    def list_available_models(self) -> List[Dict]:
        """List all available models."""
        try:
            response = requests.get(f"{self.api_url}/tags")
            if response.status_code == 200:
                data = response.json()
                return data.get('models', [])
            else:
                logger.error(f"Failed to list models: HTTP {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available locally."""
        models = self.list_available_models()
        return any(model['name'] == model_name for model in models)
    
    def pull_model(self, model_name: str, timeout: int = 600) -> bool:
        """Pull a model from the Ollama repository."""
        if self.is_model_available(model_name):
            logger.info(f"Model {model_name} is already available")
            return True
        
        logger.info(f"Pulling model: {model_name} (this may take several minutes...)")
        
        try:
            # Use subprocess for better progress handling
            result = subprocess.run(
                ['ollama', 'pull', model_name],
                timeout=timeout,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"✓ Successfully pulled model: {model_name}")
                return True
            else:
                logger.error(f"Failed to pull model {model_name}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout pulling model {model_name} (>{timeout}s)")
            return False
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    def pull_required_models(self, include_optional: bool = False) -> Dict[str, bool]:
        """Pull all required models."""
        results = {}
        
        for model_key, config in REQUIRED_MODELS.items():
            if not config['recommended'] and not include_optional:
                continue
                
            model_name = config['name']
            logger.info(f"\\nProcessing {model_key}: {model_name}")
            logger.info(f"Description: {config['description']}")
            logger.info(f"Size: {config['size']}")
            
            success = self.pull_model(model_name)
            results[model_name] = success
            
            if not success:
                logger.warning(f"Failed to pull {model_name}")
        
        return results
    
    def validate_models(self) -> Dict[str, bool]:
        """Validate that required models are available and working."""
        logger.info("Validating Ollama models...")
        results = {}
        
        for model_key, config in REQUIRED_MODELS.items():
            if not config['recommended']:
                continue
                
            model_name = config['name']
            
            # Check availability
            if not self.is_model_available(model_name):
                logger.error(f"❌ Model {model_name} is not available")
                results[model_name] = False
                continue
            
            # Test model functionality
            try:
                logger.info(f"Testing model: {model_name}")
                
                if 'embed' in model_name:
                    # Test embedding model
                    success = self._test_embedding_model(model_name)
                else:
                    # Test LLM model
                    success = self._test_llm_model(model_name)
                
                if success:
                    logger.info(f"✓ Model {model_name} is working correctly")
                else:
                    logger.error(f"❌ Model {model_name} failed functionality test")
                
                results[model_name] = success
                
            except Exception as e:
                logger.error(f"❌ Error testing model {model_name}: {e}")
                results[model_name] = False
        
        return results
    
    def _test_embedding_model(self, model_name: str) -> bool:
        """Test embedding model functionality."""
        try:
            payload = {
                'model': model_name,
                'prompt': 'This is a test sentence for embedding generation.'
            }
            
            response = requests.post(
                f"{self.api_url}/embeddings",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                embedding = data.get('embedding', [])
                return len(embedding) > 0
            
            return False
            
        except Exception as e:
            logger.debug(f"Embedding test error: {e}")
            return False
    
    def _test_llm_model(self, model_name: str) -> bool:
        """Test LLM model functionality."""
        try:
            payload = {
                'model': model_name,
                'prompt': 'Say "Hello" if you can understand this message.',
                'stream': False
            }
            
            response = requests.post(
                f"{self.api_url}/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                generated_text = data.get('response', '').strip().lower()
                return 'hello' in generated_text
            
            return False
            
        except Exception as e:
            logger.debug(f"LLM test error: {e}")
            return False
    
    def optimize_performance(self, profile: str = 'balanced') -> bool:
        """Apply performance optimizations."""
        if profile not in PERFORMANCE_CONFIGS:
            logger.error(f"Unknown performance profile: {profile}")
            return False
        
        config = PERFORMANCE_CONFIGS[profile]
        logger.info(f"Applying performance profile: {profile}")
        logger.info(f"Description: {config['description']}")
        
        # Set environment variables
        env_vars = {k: str(v) for k, v in config.items() if k != 'description'}
        
        for var, value in env_vars.items():
            os.environ[var] = value
            logger.info(f"Set {var}={value}")
        
        # Save to a configuration file for persistence
        config_dir = Path.home() / '.ollama'
        config_dir.mkdir(exist_ok=True)
        
        config_file = config_dir / 'environment'
        with open(config_file, 'w') as f:
            for var, value in env_vars.items():
                f.write(f"export {var}={value}\\n")
        
        logger.info(f"Configuration saved to: {config_file}")
        logger.info("Restart Ollama service for changes to take effect")
        
        return True
    
    def show_status(self) -> None:
        """Show comprehensive Ollama status."""
        logger.info("\\n" + "="*60)
        logger.info("OLLAMA STATUS REPORT")
        logger.info("="*60)
        
        # Installation status
        installed = self.is_ollama_installed()
        running = self.is_ollama_running()
        
        logger.info(f"Installed: {'✓' if installed else '❌'}")
        logger.info(f"Running:   {'✓' if running else '❌'}")
        
        if not installed:
            logger.info("\\nTo install: python setup_ollama.py --install")
            return
        
        if not running:
            logger.info("\\nTo start: python setup_ollama.py --start")
            return
        
        # Model status
        logger.info("\\nMODEL STATUS:")
        available_models = self.list_available_models()
        
        for model_key, config in REQUIRED_MODELS.items():
            model_name = config['name']
            is_available = any(m['name'] == model_name for m in available_models)
            status = "✓" if is_available else "❌"
            priority = "REQUIRED" if config['recommended'] else "OPTIONAL"
            
            logger.info(f"  {status} {model_name:20} [{priority}] - {config['description']}")
        
        # Resource usage (if available)
        try:
            response = requests.get(f"{self.api_url}/ps")
            if response.status_code == 200:
                data = response.json()
                loaded_models = data.get('models', [])
                
                if loaded_models:
                    logger.info("\\nLOADED MODELS:")
                    for model in loaded_models:
                        name = model.get('name', 'Unknown')
                        size = model.get('size', 0)
                        size_mb = size / (1024 * 1024) if size else 0
                        logger.info(f"  • {name} ({size_mb:.1f} MB)")
                else:
                    logger.info("\\nNo models currently loaded in memory")
        except Exception:
            pass
        
        logger.info("\\nFor model management:")
        logger.info("  Pull models:     python setup_ollama.py --pull-models")
        logger.info("  Validate models: python setup_ollama.py --validate")
        logger.info("  Optimize:        python setup_ollama.py --optimize")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ollama setup and management for MCP Crawl4AI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--install',
        action='store_true',
        help='Install Ollama'
    )
    
    parser.add_argument(
        '--start',
        action='store_true',
        help='Start Ollama service'
    )
    
    parser.add_argument(
        '--pull-models',
        action='store_true',
        help='Pull required models'
    )
    
    parser.add_argument(
        '--include-optional',
        action='store_true',
        help='Include optional models when pulling'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate model functionality'
    )
    
    parser.add_argument(
        '--optimize',
        choices=list(PERFORMANCE_CONFIGS.keys()),
        help='Apply performance optimization profile'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show Ollama status'
    )
    
    parser.add_argument(
        '--base-url',
        default='http://localhost:11434',
        help='Ollama base URL'
    )
    
    args = parser.parse_args()
    
    # Create manager
    manager = OllamaManager(args.base_url)
    
    success = True
    
    try:
        if args.install:
            success &= manager.install_ollama()
            success &= manager.start_ollama_service()
        
        if args.start:
            success &= manager.start_ollama_service()
        
        if args.pull_models:
            if not manager.is_ollama_running():
                logger.error("Ollama service is not running. Start it first with --start")
                return 1
            
            results = manager.pull_required_models(args.include_optional)
            success &= all(results.values())
        
        if args.validate:
            if not manager.is_ollama_running():
                logger.error("Ollama service is not running. Start it first with --start")
                return 1
            
            results = manager.validate_models()
            success &= all(results.values())
        
        if args.optimize:
            success &= manager.optimize_performance(args.optimize)
        
        if args.status or not any([args.install, args.start, args.pull_models, 
                                  args.validate, args.optimize]):
            manager.show_status()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())