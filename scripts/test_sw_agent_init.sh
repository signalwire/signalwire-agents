#!/bin/bash
# test_sw_agent_init.sh - Comprehensive test for sw-agent-init generated projects
#
# Tests all combinations of:
#   - Agent types: basic, full
#   - Platforms: local, aws, gcp, azure
#
# Each generated project is tested with swaig-test for:
#   - --list-tools
#   - --dump-swml
#   - --exec get_info --topic "test"
#   - Serverless simulation (where applicable)

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Test directory
TEST_DIR=$(mktemp -d)
echo "Test directory: $TEST_DIR"

cleanup() {
    echo ""
    echo "Cleaning up $TEST_DIR"
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

echo "========================================"
echo "  sw-agent-init Test Suite"
echo "========================================"
echo ""

# Helper functions
log_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

log_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
    TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
}

log_section() {
    echo ""
    echo "----------------------------------------"
    echo -e "${YELLOW}$1${NC}"
    echo "----------------------------------------"
}

# Run swaig-test with PORT=3000
run_swaig_test() {
    PORT=3000 swaig-test "$@"
}

# Test function for a generated project
test_project() {
    local project_name=$1
    local project_path=$2
    local platform=$3
    local agent_file=$4

    # Determine the correct file to test based on platform
    local test_file="$project_path/$agent_file"

    if [ ! -f "$test_file" ]; then
        log_fail "$project_name: Test file not found: $test_file"
        return 1
    fi

    # Test --list-tools
    log_test "$project_name: --list-tools"
    if run_swaig_test "$test_file" --list-tools >/dev/null 2>&1; then
        log_pass "$project_name: --list-tools"
    else
        log_fail "$project_name: --list-tools"
        return 1
    fi

    # Test --dump-swml
    log_test "$project_name: --dump-swml"
    local swml_output
    swml_output=$(run_swaig_test "$test_file" --dump-swml --raw 2>&1) || true
    if echo "$swml_output" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'version' in d and 'sections' in d" 2>/dev/null; then
        log_pass "$project_name: --dump-swml"
    else
        log_fail "$project_name: --dump-swml (invalid SWML structure)"
        echo "Output was: $swml_output" | head -5
        return 1
    fi

    # Test --exec get_info
    log_test "$project_name: --exec get_info"
    if run_swaig_test "$test_file" --exec get_info --topic "test" >/dev/null 2>&1; then
        log_pass "$project_name: --exec get_info"
    else
        log_fail "$project_name: --exec get_info"
        return 1
    fi

    return 0
}

# Test serverless simulation
test_serverless_simulation() {
    local project_name=$1
    local project_path=$2
    local platform=$3
    local agent_file=$4

    local test_file="$project_path/$agent_file"

    case $platform in
        aws)
            log_test "$project_name: Lambda simulation"
            if run_swaig_test "$test_file" --simulate-serverless lambda --dump-swml --raw >/dev/null 2>&1; then
                log_pass "$project_name: Lambda simulation"
            else
                log_fail "$project_name: Lambda simulation"
            fi
            ;;
        gcp)
            log_test "$project_name: Cloud Function simulation"
            if run_swaig_test "$test_file" --simulate-serverless cloud_function --dump-swml --raw >/dev/null 2>&1; then
                log_pass "$project_name: Cloud Function simulation"
            else
                log_fail "$project_name: Cloud Function simulation"
            fi
            ;;
        azure)
            log_test "$project_name: Azure Function simulation"
            if run_swaig_test "$test_file" --simulate-serverless azure_function --dump-swml --raw >/dev/null 2>&1; then
                log_pass "$project_name: Azure Function simulation"
            else
                log_fail "$project_name: Azure Function simulation"
            fi
            ;;
        local)
            # Test all serverless simulations for local projects
            log_test "$project_name: Lambda simulation"
            if run_swaig_test "$test_file" --simulate-serverless lambda --dump-swml --raw >/dev/null 2>&1; then
                log_pass "$project_name: Lambda simulation"
            else
                log_fail "$project_name: Lambda simulation"
            fi

            log_test "$project_name: CGI simulation"
            if run_swaig_test "$test_file" --simulate-serverless cgi --cgi-host example.com --dump-swml --raw >/dev/null 2>&1; then
                log_pass "$project_name: CGI simulation"
            else
                log_fail "$project_name: CGI simulation"
            fi

            log_test "$project_name: Cloud Function simulation"
            if run_swaig_test "$test_file" --simulate-serverless cloud_function --dump-swml --raw >/dev/null 2>&1; then
                log_pass "$project_name: Cloud Function simulation"
            else
                log_fail "$project_name: Cloud Function simulation"
            fi

            log_test "$project_name: Azure Function simulation"
            if run_swaig_test "$test_file" --simulate-serverless azure_function --dump-swml --raw >/dev/null 2>&1; then
                log_pass "$project_name: Azure Function simulation"
            else
                log_fail "$project_name: Azure Function simulation"
            fi
            ;;
    esac
}

# Verify project structure
verify_structure() {
    local project_name=$1
    local project_path=$2
    local platform=$3
    local type=$4

    log_test "$project_name: Checking project structure"
    local structure_ok=true

    # Platform-specific structure checks
    case $platform in
        local)
            # Local has full project structure
            for file in "agents/main_agent.py" "agents/__init__.py" "skills/__init__.py" ".env" ".env.example" ".gitignore" "requirements.txt" "README.md" "app.py"; do
                if [ ! -f "$project_path/$file" ]; then
                    log_fail "$project_name: Missing $file"
                    structure_ok=false
                fi
            done
            # Type-specific files
            if [ "$type" = "full" ]; then
                [ -d "$project_path/web" ] || { log_fail "$project_name: Missing web/ directory"; structure_ok=false; }
            fi
            ;;
        aws)
            # AWS has simplified serverless structure
            for file in "handler.py" "requirements.txt" "deploy.sh" ".env.example" ".gitignore" "README.md"; do
                if [ ! -f "$project_path/$file" ]; then
                    log_fail "$project_name: Missing $file"
                    structure_ok=false
                fi
            done
            ;;
        gcp)
            # GCP has simplified serverless structure
            for file in "main.py" "requirements.txt" "deploy.sh" ".env.example" ".gitignore" "README.md"; do
                if [ ! -f "$project_path/$file" ]; then
                    log_fail "$project_name: Missing $file"
                    structure_ok=false
                fi
            done
            ;;
        azure)
            # Azure has function app structure
            for file in "function_app/__init__.py" "requirements.txt" "deploy.sh" "host.json" ".env.example" ".gitignore" "README.md"; do
                if [ ! -f "$project_path/$file" ]; then
                    log_fail "$project_name: Missing $file"
                    structure_ok=false
                fi
            done
            ;;
    esac

    if $structure_ok; then
        log_pass "$project_name: Project structure valid"
    fi
}

# Generate and test a project
generate_and_test() {
    local name=$1
    local type=$2
    local platform=$3

    local project_name="${name}-${type}-${platform}"
    local project_path="$TEST_DIR/$project_name"

    log_section "Testing: $project_name (type=$type, platform=$platform)"

    # Generate the project
    log_test "$project_name: Generating project"
    if ! sw-agent-init "$project_name" --type "$type" --platform "$platform" --no-venv --dir "$TEST_DIR" >/dev/null 2>&1; then
        log_fail "$project_name: Generation failed"
        return 1
    fi
    log_pass "$project_name: Generated"

    # Determine the agent file based on platform
    local agent_file
    case $platform in
        local)
            agent_file="app.py"
            ;;
        aws)
            agent_file="handler.py"
            ;;
        gcp)
            agent_file="main.py"
            ;;
        azure)
            agent_file="function_app/__init__.py"
            ;;
    esac

    # Verify structure
    verify_structure "$project_name" "$project_path" "$platform" "$type"

    # Test the project
    test_project "$project_name" "$project_path" "$platform" "$agent_file"

    # Test serverless simulations
    test_serverless_simulation "$project_name" "$project_path" "$platform" "$agent_file"
}

# Main test matrix
TYPES=("basic" "full")
PLATFORMS=("local" "aws" "gcp" "azure")

for type in "${TYPES[@]}"; do
    for platform in "${PLATFORMS[@]}"; do
        generate_and_test "agent" "$type" "$platform"
    done
done

# Summary
echo ""
echo "========================================"
echo "  Test Summary"
echo "========================================"
echo -e "${GREEN}Passed:${NC}  $TESTS_PASSED"
echo -e "${RED}Failed:${NC}  $TESTS_FAILED"
echo -e "${YELLOW}Skipped:${NC} $TESTS_SKIPPED"
echo ""

if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
