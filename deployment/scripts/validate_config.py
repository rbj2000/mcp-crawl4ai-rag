#!/usr/bin/env python3
"""
Configuration validation script for MCP Crawl4AI RAG deployment.

This script validates:
- Environment file completeness and syntax
- Provider compatibility and requirements
- Docker Compose configuration
- Network and port configurations
- Security settings
- Performance configurations
"""

import os
import sys
import argparse
import json
import yaml
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of a validation check."""
    check_name: str
    passed: bool
    message: str
    severity: str = 'error'  # error, warning, info

class ConfigValidator:
    """Validates deployment configuration."""
    
    def __init__(self, environment: str):
        self.environment = environment
        self.project_root = Path(__file__).parent.parent.parent
        self.results: List[ValidationResult] = []
        
        # Configuration schemas
        self.provider_schemas = {
            'openai': {
                'required_vars': ['OPENAI_API_KEY', 'MODEL_CHOICE'],
                'optional_vars': ['OPENAI_EMBEDDING_MODEL', 'OPENAI_RPM_LIMIT'],
                'embedding_dimensions': [1536]
            },
            'ollama': {
                'required_vars': [
                    'OLLAMA_BASE_URL', 'OLLAMA_EMBEDDING_MODEL', 
                    'OLLAMA_LLM_MODEL', 'OLLAMA_EMBEDDING_DIMENSION'
                ],
                'optional_vars': ['OLLAMA_NUM_PARALLEL', 'OLLAMA_LOAD_TIMEOUT'],
                'embedding_dimensions': [384, 768, 1024, 4096]
            },
            'hybrid': {
                'required_vars': [
                    'OPENAI_API_KEY', 'OLLAMA_BASE_URL', 
                    'OLLAMA_LLM_MODEL', 'EMBEDDING_PROVIDER', 'LLM_PROVIDER'
                ],
                'optional_vars': ['FAILOVER_TIMEOUT', 'AUTO_FAILBACK'],
                'embedding_dimensions': [1536, 768]
            }
        }
        
        self.database_schemas = {
            'sqlite': {
                'required_vars': ['SQLITE_DB_PATH', 'EMBEDDING_DIMENSION'],
                'optional_vars': ['SQLITE_TIMEOUT']
            },
            'supabase': {
                'required_vars': ['SUPABASE_URL', 'SUPABASE_SERVICE_KEY'],
                'optional_vars': ['SUPABASE_TIMEOUT']
            },
            'neo4j_vector': {
                'required_vars': [
                    'NEO4J_URI', 'NEO4J_USER', 'NEO4J_PASSWORD', 'NEO4J_DATABASE'
                ],
                'optional_vars': ['NEO4J_MAX_CONNECTION_POOL_SIZE']
            }
        }
    
    def load_environment_config(self, env_file: str) -> Dict[str, str]:
        """Load environment configuration from file."""
        env_path = Path(env_file)
        if not env_path.exists():
            raise FileNotFoundError(f"Environment file not found: {env_file}")
        
        config = {}
        with open(env_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip comments and empty lines
                if line.startswith('#') or not line:
                    continue
                
                # Parse KEY=VALUE pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    
                    config[key] = value
                else:
                    self.add_result(ValidationResult(
                        f"syntax_line_{line_num}",
                        False,
                        f"Invalid syntax on line {line_num}: {line}",
                        'warning'
                    ))
        
        return config
    
    def add_result(self, result: ValidationResult):
        """Add validation result."""
        self.results.append(result)
        
        if result.severity == 'error':
            logger.error(f"❌ {result.check_name}: {result.message}")
        elif result.severity == 'warning':
            logger.warning(f"⚠️  {result.check_name}: {result.message}")
        else:
            logger.info(f"ℹ️  {result.check_name}: {result.message}")
    
    def validate_basic_structure(self, config: Dict[str, str]) -> bool:
        """Validate basic configuration structure."""
        required_base_vars = [
            'HOST', 'PORT', 'ENVIRONMENT', 'AI_PROVIDER', 'VECTOR_DB_PROVIDER'
        ]
        
        missing_vars = []
        for var in required_base_vars:
            if var not in config:
                missing_vars.append(var)
        
        if missing_vars:
            self.add_result(ValidationResult(
                'basic_structure',
                False,
                f"Missing required base variables: {missing_vars}"
            ))
            return False
        
        self.add_result(ValidationResult(
            'basic_structure',
            True,
            "Basic configuration structure is valid"
        ))
        return True
    
    def validate_ai_provider(self, config: Dict[str, str]) -> bool:
        """Validate AI provider configuration."""
        ai_provider = config.get('AI_PROVIDER', '').lower()
        
        if ai_provider not in self.provider_schemas:
            self.add_result(ValidationResult(
                'ai_provider',
                False,
                f"Unsupported AI provider: {ai_provider}. Supported: {list(self.provider_schemas.keys())}"
            ))
            return False
        
        schema = self.provider_schemas[ai_provider]
        missing_vars = []
        
        for var in schema['required_vars']:
            if var not in config or not config[var]:
                missing_vars.append(var)
        
        if missing_vars:
            self.add_result(ValidationResult(
                'ai_provider_config',
                False,
                f"Missing required {ai_provider} variables: {missing_vars}"
            ))
            return False
        
        # Validate provider-specific settings
        if ai_provider == 'openai':
            self._validate_openai_config(config)
        elif ai_provider == 'ollama':
            self._validate_ollama_config(config)
        elif ai_provider == 'hybrid':
            self._validate_hybrid_config(config)
        
        self.add_result(ValidationResult(
            'ai_provider',
            True,
            f"AI provider configuration ({ai_provider}) is valid"
        ))
        return True
    
    def _validate_openai_config(self, config: Dict[str, str]):
        """Validate OpenAI-specific configuration."""
        api_key = config.get('OPENAI_API_KEY', '')
        
        # Check API key format
        if not api_key.startswith('sk-'):
            self.add_result(ValidationResult(
                'openai_api_key',
                False,
                "OpenAI API key should start with 'sk-'",
                'warning'
            ))
        
        # Validate model choice
        model = config.get('MODEL_CHOICE', '')
        valid_models = ['gpt-4', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo']
        if model and model not in valid_models:
            self.add_result(ValidationResult(
                'openai_model',
                False,
                f"Potentially invalid OpenAI model: {model}",
                'warning'
            ))
    
    def _validate_ollama_config(self, config: Dict[str, str]):
        """Validate Ollama-specific configuration."""
        base_url = config.get('OLLAMA_BASE_URL', '')
        
        # Validate URL format
        if base_url:
            try:
                parsed = urlparse(base_url)
                if not parsed.scheme or not parsed.netloc:
                    self.add_result(ValidationResult(
                        'ollama_url',
                        False,
                        f"Invalid Ollama base URL format: {base_url}"
                    ))
            except Exception:
                self.add_result(ValidationResult(
                    'ollama_url',
                    False,
                    f"Invalid Ollama base URL: {base_url}"
                ))
        
        # Validate embedding dimensions
        dim_str = config.get('OLLAMA_EMBEDDING_DIMENSION', '')
        if dim_str:
            try:
                dim = int(dim_str)
                valid_dims = self.provider_schemas['ollama']['embedding_dimensions']
                if dim not in valid_dims:
                    self.add_result(ValidationResult(
                        'ollama_dimensions',
                        False,
                        f"Unusual embedding dimension: {dim}. Common values: {valid_dims}",
                        'warning'
                    ))
            except ValueError:
                self.add_result(ValidationResult(
                    'ollama_dimensions',
                    False,
                    f"Invalid embedding dimension format: {dim_str}"
                ))
    
    def _validate_hybrid_config(self, config: Dict[str, str]):
        """Validate hybrid provider configuration."""
        embedding_provider = config.get('EMBEDDING_PROVIDER', '').lower()
        llm_provider = config.get('LLM_PROVIDER', '').lower()
        
        valid_providers = ['openai', 'ollama']
        
        if embedding_provider not in valid_providers:
            self.add_result(ValidationResult(
                'hybrid_embedding_provider',
                False,
                f"Invalid embedding provider: {embedding_provider}. Valid: {valid_providers}"
            ))
        
        if llm_provider not in valid_providers:
            self.add_result(ValidationResult(
                'hybrid_llm_provider',
                False,
                f"Invalid LLM provider: {llm_provider}. Valid: {valid_providers}"
            ))
        
        # Validate that required configs for each provider are present
        if embedding_provider == 'openai' and not config.get('OPENAI_API_KEY'):
            self.add_result(ValidationResult(
                'hybrid_openai_key',
                False,
                "OpenAI API key required when using OpenAI embedding provider"
            ))
        
        if llm_provider == 'ollama' and not config.get('OLLAMA_BASE_URL'):
            self.add_result(ValidationResult(
                'hybrid_ollama_url',
                False,
                "Ollama base URL required when using Ollama LLM provider"
            ))
    
    def validate_database_provider(self, config: Dict[str, str]) -> bool:
        """Validate database provider configuration."""
        db_provider = config.get('VECTOR_DB_PROVIDER', '').lower()
        
        if db_provider not in self.database_schemas:
            self.add_result(ValidationResult(
                'database_provider',
                False,
                f"Unsupported database provider: {db_provider}. Supported: {list(self.database_schemas.keys())}"
            ))
            return False
        
        schema = self.database_schemas[db_provider]
        missing_vars = []
        
        for var in schema['required_vars']:
            if var not in config or not config[var]:
                missing_vars.append(var)
        
        if missing_vars:
            self.add_result(ValidationResult(
                'database_config',
                False,
                f"Missing required {db_provider} variables: {missing_vars}"
            ))
            return False
        
        # Database-specific validations
        if db_provider == 'supabase':
            self._validate_supabase_config(config)
        elif db_provider == 'neo4j_vector':
            self._validate_neo4j_config(config)
        
        self.add_result(ValidationResult(
            'database_provider',
            True,
            f"Database provider configuration ({db_provider}) is valid"
        ))
        return True
    
    def _validate_supabase_config(self, config: Dict[str, str]):
        """Validate Supabase-specific configuration."""
        url = config.get('SUPABASE_URL', '')
        
        if url and not url.startswith('https://'):
            self.add_result(ValidationResult(
                'supabase_url',
                False,
                "Supabase URL should use HTTPS",
                'warning'
            ))
    
    def _validate_neo4j_config(self, config: Dict[str, str]):
        """Validate Neo4j-specific configuration."""
        uri = config.get('NEO4J_URI', '')
        
        if uri and not (uri.startswith('bolt://') or uri.startswith('neo4j://')):
            self.add_result(ValidationResult(
                'neo4j_uri',
                False,
                "Neo4j URI should use bolt:// or neo4j:// protocol",
                'warning'
            ))
    
    def validate_embedding_dimensions(self, config: Dict[str, str]) -> bool:
        """Validate embedding dimension consistency."""
        ai_provider = config.get('AI_PROVIDER', '').lower()
        embedding_dim = config.get('EMBEDDING_DIMENSION')
        
        if not embedding_dim:
            self.add_result(ValidationResult(
                'embedding_dimensions',
                False,
                "EMBEDDING_DIMENSION not specified"
            ))
            return False
        
        try:
            dim = int(embedding_dim)
        except ValueError:
            self.add_result(ValidationResult(
                'embedding_dimensions',
                False,
                f"Invalid embedding dimension format: {embedding_dim}"
            ))
            return False
        
        # Check if dimension matches provider expectations
        if ai_provider in self.provider_schemas:
            expected_dims = self.provider_schemas[ai_provider]['embedding_dimensions']
            if dim not in expected_dims:
                self.add_result(ValidationResult(
                    'embedding_dimensions',
                    False,
                    f"Dimension {dim} may not be compatible with {ai_provider}. Expected: {expected_dims}",
                    'warning'
                ))
        
        # Check provider-specific dimension variables
        if ai_provider == 'ollama':
            ollama_dim = config.get('OLLAMA_EMBEDDING_DIMENSION')
            if ollama_dim and int(ollama_dim) != dim:
                self.add_result(ValidationResult(
                    'embedding_dimensions_mismatch',
                    False,
                    f"EMBEDDING_DIMENSION ({dim}) doesn't match OLLAMA_EMBEDDING_DIMENSION ({ollama_dim})"
                ))
                return False
        
        self.add_result(ValidationResult(
            'embedding_dimensions',
            True,
            f"Embedding dimensions ({dim}) are consistent"
        ))
        return True
    
    def validate_performance_config(self, config: Dict[str, str]) -> bool:
        """Validate performance-related configuration."""
        checks_passed = True
        
        # Validate numeric performance settings
        numeric_settings = {
            'MAX_CONCURRENT_REQUESTS': (1, 100),
            'BATCH_SIZE': (1, 1000),
            'CACHE_TTL': (0, 86400),  # 0 to 24 hours
            'EMBEDDINGS_BATCH_SIZE': (1, 100)
        }
        
        for setting, (min_val, max_val) in numeric_settings.items():
            value = config.get(setting)
            if value:
                try:
                    num_val = int(value)
                    if not (min_val <= num_val <= max_val):
                        self.add_result(ValidationResult(
                            f'performance_{setting.lower()}',
                            False,
                            f"{setting} ({num_val}) outside recommended range [{min_val}, {max_val}]",
                            'warning'
                        ))
                except ValueError:
                    self.add_result(ValidationResult(
                        f'performance_{setting.lower()}',
                        False,
                        f"Invalid {setting} format: {value}"
                    ))
                    checks_passed = False
        
        # Environment-specific performance validations
        if self.environment == 'production':
            self._validate_production_performance(config)
        
        return checks_passed
    
    def _validate_production_performance(self, config: Dict[str, str]):
        """Validate production-specific performance settings."""
        # Check if monitoring is enabled
        if config.get('ENABLE_METRICS', '').lower() != 'true':
            self.add_result(ValidationResult(
                'production_monitoring',
                False,
                "Metrics should be enabled in production",
                'warning'
            ))
        
        # Check log level
        log_level = config.get('LOG_LEVEL', '').upper()
        if log_level == 'DEBUG':
            self.add_result(ValidationResult(
                'production_logging',
                False,
                "DEBUG logging not recommended for production",
                'warning'
            ))
    
    def validate_security_config(self, config: Dict[str, str]) -> bool:
        """Validate security-related configuration."""
        checks_passed = True
        
        # CORS validation
        cors_origins = config.get('CORS_ORIGINS', '')
        if cors_origins == '*' and self.environment != 'development':
            self.add_result(ValidationResult(
                'security_cors',
                False,
                "Wildcard CORS origins not recommended for production",
                'warning'
            ))
        
        # Rate limiting
        rate_limit = config.get('RATE_LIMIT_PER_MINUTE')
        if rate_limit:
            try:
                limit = int(rate_limit)
                if self.environment == 'production' and limit > 1000:
                    self.add_result(ValidationResult(
                        'security_rate_limit',
                        False,
                        f"Very high rate limit ({limit}) may cause resource issues",
                        'warning'
                    ))
            except ValueError:
                self.add_result(ValidationResult(
                    'security_rate_limit',
                    False,
                    f"Invalid rate limit format: {rate_limit}"
                ))
                checks_passed = False
        
        return checks_passed
    
    def validate_docker_compose(self) -> bool:
        """Validate Docker Compose configuration."""
        compose_file = self.project_root / 'docker-compose.yml'
        
        if not compose_file.exists():
            self.add_result(ValidationResult(
                'docker_compose',
                False,
                f"Docker Compose file not found: {compose_file}"
            ))
            return False
        
        try:
            with open(compose_file, 'r') as f:
                compose_config = yaml.safe_load(f)
            
            # Validate services
            services = compose_config.get('services', {})
            if not services:
                self.add_result(ValidationResult(
                    'docker_compose_services',
                    False,
                    "No services defined in Docker Compose"
                ))
                return False
            
            # Validate volumes
            volumes = compose_config.get('volumes', {})
            expected_volumes = ['ollama_models', 'neo4j_data', 'prometheus_data']
            
            for volume in expected_volumes:
                if volume not in volumes:
                    self.add_result(ValidationResult(
                        f'docker_volume_{volume}',
                        False,
                        f"Expected volume '{volume}' not defined",
                        'info'
                    ))
            
            self.add_result(ValidationResult(
                'docker_compose',
                True,
                "Docker Compose configuration is valid"
            ))
            return True
            
        except yaml.YAMLError as e:
            self.add_result(ValidationResult(
                'docker_compose',
                False,
                f"Invalid Docker Compose YAML: {e}"
            ))
            return False
    
    def run_full_validation(self, env_file: str) -> bool:
        """Run complete validation suite."""
        logger.info(f"Starting configuration validation for environment: {self.environment}")
        
        try:
            # Load configuration
            config = self.load_environment_config(env_file)
            logger.info(f"Loaded configuration from: {env_file}")
            
            # Run all validations
            validations = [
                ('Basic Structure', lambda: self.validate_basic_structure(config)),
                ('AI Provider', lambda: self.validate_ai_provider(config)),
                ('Database Provider', lambda: self.validate_database_provider(config)),
                ('Embedding Dimensions', lambda: self.validate_embedding_dimensions(config)),
                ('Performance Settings', lambda: self.validate_performance_config(config)),
                ('Security Settings', lambda: self.validate_security_config(config)),
                ('Docker Compose', lambda: self.validate_docker_compose())
            ]
            
            all_passed = True
            for name, validation_func in validations:
                logger.info(f"Validating {name}...")
                passed = validation_func()
                if not passed:
                    all_passed = False
            
            # Summary
            self.print_summary()
            
            return all_passed
            
        except Exception as e:
            self.add_result(ValidationResult(
                'validation_error',
                False,
                f"Validation failed with error: {e}"
            ))
            return False
    
    def print_summary(self):
        """Print validation summary."""
        errors = [r for r in self.results if r.severity == 'error' and not r.passed]
        warnings = [r for r in self.results if r.severity == 'warning' and not r.passed]
        passed = [r for r in self.results if r.passed]
        
        logger.info("\n" + "="*60)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*60)
        logger.info(f"✅ Passed: {len(passed)}")
        logger.info(f"⚠️  Warnings: {len(warnings)}")
        logger.info(f"❌ Errors: {len(errors)}")
        
        if errors:
            logger.info("\nERRORS:")
            for error in errors:
                logger.error(f"  - {error.check_name}: {error.message}")
        
        if warnings:
            logger.info("\nWARNINGS:")
            for warning in warnings:
                logger.warning(f"  - {warning.check_name}: {warning.message}")
    
    def export_results(self, output_file: str):
        """Export validation results to JSON."""
        results_data = {
            'environment': self.environment,
            'validation_time': str(time.time()),
            'summary': {
                'total_checks': len(self.results),
                'passed': len([r for r in self.results if r.passed]),
                'warnings': len([r for r in self.results if r.severity == 'warning' and not r.passed]),
                'errors': len([r for r in self.results if r.severity == 'error' and not r.passed])
            },
            'results': [
                {
                    'check_name': r.check_name,
                    'passed': r.passed,
                    'message': r.message,
                    'severity': r.severity
                }
                for r in self.results
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Validation results exported to: {output_file}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate MCP Crawl4AI RAG deployment configuration"
    )
    
    parser.add_argument(
        '--env', '--environment',
        choices=['development', 'staging', 'production'],
        default='development',
        help='Environment to validate'
    )
    
    parser.add_argument(
        '--env-file',
        help='Environment file to validate'
    )
    
    parser.add_argument(
        '--output',
        help='Export results to JSON file'
    )
    
    args = parser.parse_args()
    
    # Determine environment file if not specified
    if not args.env_file:
        project_root = Path(__file__).parent.parent.parent
        args.env_file = str(project_root / 'deployment' / 'environments' / f'.env.{args.env}')
    
    # Create validator
    validator = ConfigValidator(args.env)
    
    try:
        # Run validation
        success = validator.run_full_validation(args.env_file)
        
        # Export results if requested
        if args.output:
            validator.export_results(args.output)
        
        if success:
            logger.info("🎉 Configuration validation passed!")
            return 0
        else:
            logger.error("💥 Configuration validation failed!")
            return 1
            
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())