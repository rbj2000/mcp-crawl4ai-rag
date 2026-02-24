#!/usr/bin/env python3
"""
Validation script for reranking configuration
Checks environment variables and provider initialization
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / 'src'
sys.path.append(str(src_path))

def load_environment():
    """Load environment variables"""
    # Try multiple environment files
    env_files = [
        project_root / '.env.local',
        project_root / '.env',
        project_root / '.env.example'
    ]
    
    for env_file in env_files:
        if env_file.exists():
            print(f"Loading environment from: {env_file}")
            load_dotenv(env_file, override=True)
            return env_file
    
    print("No environment file found")
    return None

def validate_reranking_env_vars():
    """Validate reranking environment variables"""
    print("\n=== Environment Variables ===")
    
    required_vars = {
        'USE_RERANKING': os.getenv('USE_RERANKING', 'false'),
        'RERANKING_PROVIDER': os.getenv('RERANKING_PROVIDER', 'NOT_SET'),
        'RERANKING_MODEL': os.getenv('RERANKING_MODEL', 'NOT_SET')
    }
    
    optional_vars = {
        'RERANKING_MAX_RESULTS': os.getenv('RERANKING_MAX_RESULTS', '100'),
        'RERANKING_TIMEOUT': os.getenv('RERANKING_TIMEOUT', '30.0'),
        'HUGGINGFACE_TOKEN': os.getenv('HUGGINGFACE_TOKEN', 'NOT_SET')
    }
    
    print("Required variables:")
    for var, value in required_vars.items():
        status = "✓" if value != 'NOT_SET' else "✗"
        print(f"  {status} {var}: {value}")
    
    print("\nOptional variables:")
    for var, value in optional_vars.items():
        status = "✓" if value != 'NOT_SET' else "-"
        print(f"  {status} {var}: {value}")
    
    return required_vars['USE_RERANKING'].lower() == 'true'

def validate_provider_config():
    """Validate AI provider configuration"""
    print("\n=== Provider Configuration ===")
    
    try:
        from ai_providers.config import AIProviderConfig
        
        config = AIProviderConfig.from_env()
        print(f"✓ Provider: {config.provider.value}")
        
        reranking_config = config.get_reranking_config()
        print(f"✓ Reranking config loaded: {len(reranking_config)} settings")
        
        for key, value in reranking_config.items():
            print(f"    {key}: {value}")
        
        return True, config
        
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False, None

def validate_provider_creation(config):
    """Validate reranking provider creation"""
    print("\n=== Provider Creation ===")
    
    try:
        from ai_providers.factory import AIProviderFactory
        from ai_providers.base import AIProvider
        
        reranking_provider_name = os.getenv('RERANKING_PROVIDER', 'huggingface').lower()
        print(f"Testing {reranking_provider_name} provider creation...")
        
        if reranking_provider_name == 'ollama':
            provider = AIProviderFactory.create_provider(
                AIProvider.OLLAMA,
                config.config
            )
        elif reranking_provider_name == 'openai':
            provider = AIProviderFactory.create_provider(
                AIProvider.OPENAI,
                config.config
            )
        elif reranking_provider_name == 'huggingface':
            provider = AIProviderFactory.create_reranking_provider(
                AIProvider.HUGGINGFACE,
                config.get_reranking_config()
            )
        else:
            print(f"✗ Unsupported provider: {reranking_provider_name}")
            return False
        
        if hasattr(provider, 'rerank_results'):
            print(f"✓ {reranking_provider_name.capitalize()} provider supports reranking")
            
            if hasattr(provider, 'get_reranking_models'):
                models = provider.get_reranking_models()
                print(f"✓ Available models: {', '.join(models)}")
            
            return True
        else:
            print(f"✗ {reranking_provider_name.capitalize()} provider does not support reranking")
            return False
            
    except Exception as e:
        print(f"✗ Provider creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main validation function"""
    print("🔍 Reranking Configuration Validation")
    print("=" * 40)
    
    # Load environment
    env_file = load_environment()
    if not env_file:
        print("❌ No environment file found")
        return False
    
    # Validate environment variables
    reranking_enabled = validate_reranking_env_vars()
    if not reranking_enabled:
        print("\n⚠️  Reranking is disabled (USE_RERANKING=false)")
        print("   Set USE_RERANKING=true to enable reranking functionality")
        return True
    
    # Validate provider configuration
    config_valid, config = validate_provider_config()
    if not config_valid:
        print("\n❌ Provider configuration validation failed")
        return False
    
    # Validate provider creation
    provider_valid = validate_provider_creation(config)
    if not provider_valid:
        print("\n❌ Provider creation validation failed")
        return False
    
    print("\n" + "=" * 40)
    print("✅ All reranking configuration validation passed!")
    print("🚀 Reranking is properly configured and ready to use")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)