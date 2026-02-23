# Lesson 6: Running and Testing Fred

Time to bring Fred to life! This lesson covers running Fred locally and testing his capabilities using curl commands.

## Table of Contents

1. [Running Fred](#running-fred)
2. [Understanding the Endpoints](#understanding-the-endpoints)
3. [Testing with Curl](#testing-with-curl)
4. [Managing Fred with Scripts](#managing-fred-with-scripts)
5. [Troubleshooting](#troubleshooting)

---

## Running Fred

Let's start Fred and see him in action!

### Step 1: Run Fred Directly

```bash
python fred.py
```

You'll see output like this:

```
============================================================
🤖 Fred - The Wikipedia Knowledge Bot
============================================================

Fred is a friendly assistant who loves searching Wikipedia!
He can help you learn about almost any topic.

Example questions you can ask Fred:
  • 'Tell me about Albert Einstein'
  • 'What is quantum physics?'
  • 'Who was Marie Curie?'
  • 'Search for information about the solar system'
  • 'Can you share a fun fact?'

Fred is available at: http://localhost:3000/fred
Basic Auth: fred_user:a7b9c2d4e6

Fred's capabilities:
  ✓ Wikipedia search (via skill)
  ✓ Fun facts about Wikipedia (custom function)

Starting Fred... Press Ctrl+C to stop.
============================================================
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:3000 (Press CTRL+C to quit)
```

### Important Information

**Note these details:**
- **URL**: `http://localhost:3000/fred`
- **Auth**: Username and password (randomly generated each time)
- **Process ID**: Shows in the server output

## Understanding the Endpoints

Fred exposes two main HTTP endpoints:

### 1. SWML Endpoint (GET /fred)

Returns Fred's configuration as a SWML (SignalWire Markup Language) document:
- **Purpose**: Tells SignalWire how Fred should behave
- **Method**: GET
- **Auth**: Required

### 2. SWAIG Endpoint (POST /fred/swaig/)

SWAIG (SignalWire AI Gateway) handles function execution:
- **Purpose**: Executes Fred's functions
- **Method**: POST
- **Auth**: Required
- **Content-Type**: application/json

## Testing with swaig-test CLI

The SignalWire SDK includes a CLI testing tool called `swaig-test`. Let's use it to explore and test Fred without running the server!

### Using swaig-test to Explore Fred

#### Discover Fred in the File

```bash
# Discover agents in fred.py
swaig-test fred.py
```

Output:
```
Available agents in fred.py:

  FredTheWikiBot
    Type: Ready instance
    Name: Fred
    Route: /fred
    Functions: 2 (search_wiki, share_fun_fact)

To use this agent:
  swaig-test fred.py --list-tools
  swaig-test fred.py --dump-swml
```

#### List Fred's Functions

```bash
# List all available functions
swaig-test fred.py --list-tools
```

Output:
```
Available SWAIG functions:
  search_wiki - Search Wikipedia for information about a topic and get article summaries
    Parameters:
      query (string) (required): The search term or topic to look up on Wikipedia
      
  share_fun_fact - Share an interesting fact about Wikipedia itself
    Parameters:
      category (string): Type of fact to share
        Allowed values: statistics, history, records, random
```

#### Get Detailed Function Info

```bash
# Verbose listing shows full schemas
swaig-test fred.py --list-tools --verbose
```

#### Test Functions Without Running the Server

```bash
# Test Wikipedia search
swaig-test fred.py --exec search_wiki --query "Python programming language"

# Test fun fact with category
swaig-test fred.py --exec share_fun_fact --category history

# Test fun fact without parameters (uses default)
swaig-test fred.py --exec share_fun_fact
```

#### Generate and Inspect SWML

```bash
# Generate SWML document
swaig-test fred.py --dump-swml

# Get raw JSON for processing
swaig-test fred.py --dump-swml --raw | jq '.sections.main[0].ai.SWAIG.functions'

# See what call data is generated
swaig-test fred.py --dump-swml --verbose
```

### Advanced swaig-test Features

#### Test with Different Call Scenarios

```bash
# Simulate inbound SIP call
swaig-test fred.py --dump-swml --call-type sip --call-direction inbound

# Custom caller info
swaig-test fred.py --dump-swml --call-from "+15551234567" --call-to "+15559876543"
```

#### Test Serverless Deployment

```bash
# Test Lambda deployment
swaig-test fred.py --simulate-serverless lambda --dump-swml

# Test with Lambda function execution
swaig-test fred.py --simulate-serverless lambda --exec search_wiki --query "SignalWire"

# Test CGI deployment
swaig-test fred.py --simulate-serverless cgi --cgi-host example.com --dump-swml
```

#### Debug Mode

```bash
# See detailed execution trace
swaig-test fred.py --exec search_wiki --query "Albert Einstein" --verbose

# Test with custom post data
swaig-test fred.py --exec share_fun_fact --post-data '{"call_id": "custom-123"}'
```

### Benefits of swaig-test

1. **No Server Required**: Test functions without running Fred
2. **Quick Validation**: Verify functions work before deployment
3. **SWML Inspection**: See exactly what SignalWire receives
4. **Serverless Testing**: Validate Lambda/CGI deployments
5. **Debugging**: Detailed traces for troubleshooting

## Testing with Curl

Once Fred is running, you can also test with curl commands. Replace `username:password` with your actual credentials from the output.

### Step 2: Fetch Fred's SWML Configuration

```bash
# Get Fred's SWML document
curl -u username:password http://localhost:3000/fred | python -m json.tool
```

**Example with real credentials:**
```bash
curl -u fred_user:a7b9c2d4e6 http://localhost:3000/fred | python -m json.tool
```

**Expected Response (shortened):**
```json
{
  "version": "1.0.0",
  "sections": {
    "main": [
      {
        "ai": {
          "prompt": {
            "pom": [
              {
                "title": "Personality",
                "body": "You are Fred, a friendly and knowledgeable assistant..."
              }
            ]
          },
          "SWAIG": {
            "functions": [
              {
                "function": "search_wiki",
                "description": "Search Wikipedia for information about a topic"
              },
              {
                "function": "share_fun_fact",
                "description": "Share an interesting fact about Wikipedia itself"
              }
            ]
          }
        }
      }
    ]
  }
}
```

### Step 3: Test Wikipedia Search

Test Fred's Wikipedia search capability:

```bash
# Search for FreeSWITCH
curl -X POST -u username:password \
  -H "Content-Type: application/json" \
  -d '{
    "function": "search_wiki",
    "argument": {
      "parsed": [
        {
          "query": "FreeSWITCH"
        }
      ]
    },
    "call_id": "test-call-123",
    "project_id": "test-project",
    "space_id": "test-space"
  }' \
  http://localhost:3000/fred/swaig/
```

**Expected Response:**
```json
{
  "response": "==================================================**FreeSWITCH**\n\nFreeSWITCH is a free and open-source telephony software for real-time communication protocols..."
}
```

### Step 4: Test Fun Fact Function

Test Fred's fun fact feature:

```bash
# Get a random fun fact
curl -X POST -u username:password \
  -H "Content-Type: application/json" \
  -d '{
    "function": "share_fun_fact",
    "argument": {
      "parsed": [{}]
    },
    "call_id": "test-call-456"
  }' \
  http://localhost:3000/fred/swaig/
```

**With category parameter:**
```bash
# Get a history fact
curl -X POST -u username:password \
  -H "Content-Type: application/json" \
  -d '{
    "function": "share_fun_fact",
    "argument": {
      "parsed": [
        {
          "category": "history"
        }
      ]
    },
    "call_id": "test-call-789"
  }' \
  http://localhost:3000/fred/swaig/
```

**Expected Response:**
```json
{
  "response": "Here's a history fact about Wikipedia: Wikipedia was launched on January 15, 2001!"
}
```

## Managing Fred with Scripts

For easier management, let's create a simple bash script to start/stop Fred:

### Step 5: Create Management Script

Create `fred.sh`:

```bash
#!/bin/bash
# Fred Bot Manager - Start/Stop Fred the Wikipedia Bot

PID_FILE="fred.pid"
LOG_FILE="fred.log"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if Fred is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# Start Fred
start_fred() {
    if is_running; then
        echo -e "${YELLOW}Fred is already running${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Starting Fred...${NC}"
    nohup python fred.py > "$LOG_FILE" 2>&1 &
    PID=$!
    echo $PID > "$PID_FILE"
    
    sleep 2
    
    if is_running; then
        echo -e "${GREEN}✅ Fred started successfully!${NC}"
        echo "   PID: $PID"
        echo "   Logs: $LOG_FILE"
        
        # Extract auth from log
        AUTH=$(grep "Basic Auth:" "$LOG_FILE" | tail -1)
        if [ ! -z "$AUTH" ]; then
            echo "   $AUTH"
        fi
    else
        echo -e "${RED}❌ Failed to start Fred${NC}"
        return 1
    fi
}

# Stop Fred
stop_fred() {
    if ! is_running; then
        echo -e "${YELLOW}Fred is not running${NC}"
        return 1
    fi
    
    PID=$(cat "$PID_FILE")
    echo -e "${GREEN}Stopping Fred...${NC}"
    kill $PID
    rm -f "$PID_FILE"
    echo -e "${GREEN}✅ Fred stopped${NC}"
}

# Main script
case "$1" in
    start)
        start_fred
        ;;
    stop)
        stop_fred
        ;;
    restart)
        stop_fred
        sleep 1
        start_fred
        ;;
    status)
        if is_running; then
            echo -e "${GREEN}● Fred is running${NC}"
            PID=$(cat "$PID_FILE")
            echo "   PID: $PID"
        else
            echo -e "${RED}● Fred is not running${NC}"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
```

Make it executable:
```bash
chmod +x fred.sh
```

### Using the Management Script

```bash
# Start Fred in background
./fred.sh start

# Check status
./fred.sh status

# View logs
tail -f fred.log

# Stop Fred
./fred.sh stop
```

## Troubleshooting

### Common Issues and Solutions

#### Port Already in Use

**Error:**
```
ERROR: [Errno 48] Address already in use
```

**Solution:**
```bash
# Find process using port 3000
lsof -i :3000

# Kill the process
kill -9 <PID>
```

#### Authentication Failed

**Error:**
```json
{"error": "Unauthorized"}
```

**Solution:**
- Check username and password from Fred's output
- Ensure you're using the correct credentials
- Don't forget the `-u` flag in curl

#### Module Not Found

**Error:**
```
ModuleNotFoundError: No module named 'signalwire_agents'
```

**Solution:**
```bash
# Install the SDK
pip install signalwire-agents

# Or if using virtual environment
source fred-env/bin/activate
pip install signalwire-agents
```

#### No Response from Functions

**Check:**
1. Function name matches exactly (`search_wiki`, not `search_wikipedia`)
2. JSON structure is correct (note the `parsed` array)
3. Content-Type header is set

### Debug Tips

#### Enable Verbose Logging

Add to Fred's `__init__`:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Test Minimal SWAIG Call

```bash
# Simplest possible function call
curl -X POST -u username:password \
  -H "Content-Type: application/json" \
  -d '{"function":"share_fun_fact","argument":{"parsed":[{}]}}' \
  http://localhost:3000/fred/swaig/
```

#### Check Fred's Health

```bash
# Simple auth check
curl -I -u username:password http://localhost:3000/fred
```

Should return:
```
HTTP/1.1 200 OK
```

## Production Considerations

### Security

1. **Use Environment Variables for Auth**
   ```bash
   export SWML_BASIC_AUTH_USER="fred_prod"
   export SWML_BASIC_AUTH_PASSWORD="strong_password_here"
   ```

2. **Use HTTPS in Production**
   - Deploy behind a reverse proxy (nginx, Apache)
   - Or use SignalWire's SSL support

### Deployment Options

1. **Direct Deployment**
   ```bash
   # Run with specific host/port
   python fred.py --host 0.0.0.0 --port 8080
   ```

2. **Docker Container**
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY fred.py .
   CMD ["python", "fred.py"]
   ```

3. **Process Manager (PM2)**
   ```bash
   pm2 start fred.py --name fred-bot
   ```

## Summary

Congratulations! You've successfully:
- ✅ Built Fred from scratch
- ✅ Run Fred locally
- ✅ Tested both Wikipedia search and fun facts
- ✅ Created management scripts
- ✅ Learned troubleshooting techniques

Fred is now ready to handle voice calls through SignalWire!

## Next Steps

To use Fred with actual phone calls:
1. Sign up for a SignalWire account
2. Configure a phone number
3. Point it to your Fred instance
4. Make Fred accessible via public URL (ngrok, deployment, etc.)

---

**Quick Reference:**

```bash
# Start Fred
python fred.py

# Test SWML
curl -u user:pass http://localhost:3000/fred

# Test Wikipedia search
curl -X POST -u user:pass -H "Content-Type: application/json" \
  -d '{"function":"search_wiki","argument":{"parsed":[{"query":"Python"}]}}' \
  http://localhost:3000/fred/swaig/

# Test fun fact
curl -X POST -u user:pass -H "Content-Type: application/json" \
  -d '{"function":"share_fun_fact","argument":{"parsed":[{}]}}' \
  http://localhost:3000/fred/swaig/
```

---

[← Previous: Custom Functions](05-custom-functions.md) | [Back to Overview](README.md) | [Complete Code →](appendix-complete-code.md)