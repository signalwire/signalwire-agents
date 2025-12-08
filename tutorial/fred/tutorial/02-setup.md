# Lesson 2: Setting Up Your Environment

Let's get your development environment ready to build Fred! This lesson covers installing the SignalWire AI Agents SDK and verifying everything works correctly.

## Table of Contents

1. [Prerequisites Check](#prerequisites-check)
2. [Installing the SDK](#installing-the-sdk)
3. [Verifying Installation](#verifying-installation)
4. [Understanding Dependencies](#understanding-dependencies)
5. [Project Structure](#project-structure)

---

## Prerequisites Check

Before we begin, ensure you have:

### Required Software

**Python 3.7+**

Check your Python version:
```bash
python3 --version
# or
python --version
```

You should see something like:
```
Python 3.8.10
```

**pip (Python Package Manager)**

Verify pip is installed:
```bash
pip3 --version
# or
pip --version
```

### Recommended Setup

**Virtual Environment (Recommended)**

It's best practice to use a virtual environment:

```bash
# Create a virtual environment
python3 -m venv fred-env

# Activate it
# On Linux/Mac:
source fred-env/bin/activate

# On Windows:
fred-env\Scripts\activate
```

## Installing the SDK

Now let's install the SignalWire AI Agents SDK:

### Basic Installation

```bash
pip install signalwire-agents
```

This installs the core SDK with all required dependencies.

### What Gets Installed

The installation includes:
- `signalwire-agents` - The main SDK
- `fastapi` - Web framework for HTTP endpoints
- `uvicorn` - ASGI server to run your agent
- `pydantic` - Data validation
- `requests` - HTTP client for API calls
- Other supporting libraries

## Verifying Installation

Let's verify everything installed correctly:

### Test Import

Create a test file called `test_install.py`:

```python
#!/usr/bin/env python3
"""Test SignalWire SDK installation"""

try:
    from signalwire_agents import AgentBase
    print("✅ SignalWire SDK imported successfully!")
    
    # Check version
    import signalwire_agents
    print(f"   Version: {signalwire_agents.__version__}")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("   Please run: pip install signalwire-agents")
```

Run it:
```bash
python test_install.py
```

Expected output:
```
✅ SignalWire SDK imported successfully!
   Version: 1.0.12
```

### Test Basic Agent

Let's create a minimal agent to ensure everything works:

```python
#!/usr/bin/env python3
"""Minimal test agent"""

from signalwire_agents import AgentBase

# Create a minimal agent
agent = AgentBase("Test Agent", route="/test")

# If this runs without errors, installation is good!
print("✅ Basic agent creation works!")
print(f"   Agent name: {agent.get_name()}")
print(f"   Agent route: /test")
```

## Understanding Dependencies

The SDK has several dependency categories:

### Core Dependencies

**Always Installed:**
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation
- `PyYAML` - YAML support
- `requests` - HTTP requests
- `beautifulsoup4` - HTML parsing (for web search skill)

### Optional Dependencies

**Search Features** (we won't need these for Fred):
```bash
# If you wanted search features:
pip install signalwire-agents[search]
```

**Development Tools** (optional but helpful):
```bash
# For testing
pip install pytest

# For code formatting
pip install black

# For linting
pip install pylint
```

## Project Structure

Let's set up a clean project structure for Fred:

### Create Project Directory

```bash
# Create Fred's directory
mkdir fred-bot
cd fred-bot

# Create subdirectories
mkdir logs
mkdir config
```

### Recommended Structure

```
fred-bot/
├── fred.py           # Main agent code
├── fred.sh           # Management script
├── requirements.txt  # Dependencies
├── config/          # Configuration files
├── logs/            # Log files
└── README.md        # Documentation
```

### Create requirements.txt

Save your dependencies for easy reinstallation:

```bash
# Generate requirements file
pip freeze > requirements.txt
```

Or create a minimal one:

```txt
signalwire-agents>=1.0.5
```

## Environment Variables

The SDK uses several optional environment variables:

### Authentication (Optional)

```bash
# Set fixed credentials (optional)
export SWML_BASIC_AUTH_USER="fred_user"
export SWML_BASIC_AUTH_PASSWORD="fred_pass123"

# Or let the SDK generate random ones
```

### Proxy Configuration (If Needed)

```bash
# If behind a proxy or using ngrok
export SWML_PROXY_URL_BASE="https://your-domain.com"
```

## Testing Your Setup

Let's create a simple test to ensure everything is ready:

```python
#!/usr/bin/env python3
"""Setup verification script"""

import sys

def check_setup():
    """Verify environment is ready for Fred"""
    
    print("🔍 Checking Fred's environment...\n")
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 7):
        print(f"✅ Python {python_version.major}.{python_version.minor} - Good!")
    else:
        print(f"❌ Python {python_version.major}.{python_version.minor} - Need 3.7+")
        return False
    
    # Check imports
    try:
        import signalwire_agents
        print("✅ SignalWire SDK - Installed")
    except ImportError:
        print("❌ SignalWire SDK - Not installed")
        return False
    
    try:
        import fastapi
        print("✅ FastAPI - Installed")
    except ImportError:
        print("❌ FastAPI - Not installed")
        return False
    
    try:
        import uvicorn
        print("✅ Uvicorn - Installed")
    except ImportError:
        print("❌ Uvicorn - Not installed")
        return False
    
    print("\n🎉 Environment is ready to build Fred!")
    return True

if __name__ == "__main__":
    if not check_setup():
        print("\n⚠️  Please install missing dependencies:")
        print("   pip install signalwire-agents")
        sys.exit(1)
```

## Common Installation Issues

### Issue: Permission Denied

```bash
# Use --user flag
pip install --user signalwire-agents

# Or use sudo (less recommended)
sudo pip install signalwire-agents
```

### Issue: pip Not Found

```bash
# Install pip
python3 -m ensurepip

# Or on Ubuntu/Debian
sudo apt-get install python3-pip
```

### Issue: SSL Certificate Errors

```bash
# Upgrade certificates
pip install --upgrade certifi

# Or temporarily (not recommended for production)
pip install --trusted-host pypi.org signalwire-agents
```

## Next Steps

Great! Your environment is ready. Let's start building Fred!

➡️ Continue to [Lesson 3: Creating Fred's Basic Structure](03-basic-agent.md)

---

**Checklist:**
- [ ] Python 3.7+ installed
- [ ] SignalWire SDK installed
- [ ] Basic import test passes
- [ ] Project directory created

---

[← Previous: Introduction](01-introduction.md) | [Back to Overview](README.md) | [Next: Basic Agent →](03-basic-agent.md)