#!/bin/bash
# Git workflow helper scripts for Milestone 1 development

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if we're in a git repository
check_git_repo() {
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log_error "Not in a git repository"
        exit 1
    fi
}

# Setup development environment
setup_dev_env() {
    log_info "Setting up development environment for Milestone 1..."
    
    # Install pre-commit if not present
    if ! command -v pre-commit &> /dev/null; then
        log_warning "pre-commit not found. Installing..."
        pip install pre-commit
    fi
    
    # Install pre-commit hooks
    pre-commit install
    pre-commit install --hook-type commit-msg
    
    # Set up git config
    git config commit.template .gitmessage
    
    log_success "Development environment setup complete!"
    log_info "You can now use conventional commits with the template"
}

# Create a new feature branch following naming conventions
create_feature_branch() {
    check_git_repo
    
    local feature_name=$1
    if [ -z "$feature_name" ]; then
        log_error "Please provide a feature name"
        echo "Usage: $0 feature <feature-name>"
        echo "Example: $0 feature churn-detection-core"
        exit 1
    fi
    
    local branch_name="feature/$feature_name"
    
    # Ensure we're on master and up to date
    git checkout master
    git pull origin master
    
    # Create and switch to feature branch
    git checkout -b "$branch_name"
    
    log_success "Created feature branch: $branch_name"
    log_info "Ready for development! Remember to use conventional commits."
}

# Create a fix branch
create_fix_branch() {
    check_git_repo
    
    local fix_name=$1
    if [ -z "$fix_name" ]; then
        log_error "Please provide a fix name"
        echo "Usage: $0 fix <fix-name>"
        echo "Example: $0 fix customer-data-validation"
        exit 1
    fi
    
    local branch_name="fix/$fix_name"
    
    # Ensure we're on master and up to date
    git checkout master
    git pull origin master
    
    # Create and switch to fix branch
    git checkout -b "$branch_name"
    
    log_success "Created fix branch: $branch_name"
}

# Smart commit function that analyzes changes and suggests commit type
smart_commit() {
    check_git_repo
    
    # Check if there are staged changes
    if ! git diff --cached --quiet; then
        log_info "Analyzing staged changes..."
        
        local staged_files=$(git diff --cached --name-only)
        local has_tests=false
        local has_docs=false
        local has_src=false
        local has_api=false
        local has_core=false
        
        # Analyze file types
        while IFS= read -r file; do
            case $file in
                *test*.py|tests/*) has_tests=true ;;
                docs/*|*.md) has_docs=true ;;
                src/tracer/api/*) has_api=true; has_src=true ;;
                src/tracer/core/*) has_core=true; has_src=true ;;
                src/*) has_src=true ;;
            esac
        done <<< "$staged_files"
        
        # Suggest commit type and scope
        local suggested_type="feat"
        local suggested_scope="churn"
        
        if $has_api; then
            suggested_scope="api"
        elif $has_core; then
            suggested_scope="core"
        fi
        
        if $has_tests && ! $has_src; then
            suggested_type="test"
        elif $has_docs && ! $has_src; then
            suggested_type="docs"
        fi
        
        log_info "Suggested commit format:"
        echo -e "${GREEN}${suggested_type}(${suggested_scope}): <your message>${NC}"
        log_info "Files to be committed:"
        echo "$staged_files"
        
        read -p "Proceed with commit? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git commit
        else
            log_info "Commit cancelled"
        fi
    else
        log_warning "No staged changes to commit"
        log_info "Use 'git add <files>' to stage changes first"
    fi
}

# Check commit message format
check_commit_format() {
    local commit_msg=$1
    if [[ ! $commit_msg =~ ^(feat|fix|docs|style|refactor|perf|test|chore)(\(.+\))?\:.+ ]]; then
        log_error "Commit message doesn't follow conventional format"
        log_info "Expected format: type(scope): description"
        log_info "Example: feat(churn): add customer behavior analyzer"
        return 1
    fi
    return 0
}

# Update branch with latest master changes
update_branch() {
    check_git_repo
    
    local current_branch=$(git branch --show-current)
    if [ "$current_branch" = "master" ]; then
        log_info "Already on master, pulling latest changes..."
        git pull origin master
    else
        log_info "Updating $current_branch with latest master changes..."
        git checkout master
        git pull origin master
        git checkout "$current_branch"
        git rebase master
        log_success "Branch $current_branch updated with master"
    fi
}

# Clean up merged branches
cleanup_branches() {
    check_git_repo
    
    log_info "Cleaning up merged branches..."
    
    # Switch to master
    git checkout master
    git pull origin master
    
    # Delete merged branches
    git branch --merged | grep -v "\*\|master\|main" | xargs -n 1 git branch -d
    
    log_success "Cleaned up merged branches"
}

# Show commit stats for Milestone 1
show_milestone_stats() {
    check_git_repo
    
    log_info "Milestone 1 Development Statistics"
    echo "=================================="
    
    # Count commits by type
    echo -e "\n${BLUE}Commits by Type:${NC}"
    git log --oneline --grep="feat(" --since="1 month ago" | wc -l | xargs echo "Features:"
    git log --oneline --grep="fix(" --since="1 month ago" | wc -l | xargs echo "Fixes:"
    git log --oneline --grep="test(" --since="1 month ago" | wc -l | xargs echo "Tests:"
    git log --oneline --grep="docs(" --since="1 month ago" | wc -l | xargs echo "Documentation:"
    
    # Count commits by scope
    echo -e "\n${BLUE}Commits by Scope:${NC}"
    git log --oneline --grep="(churn)" --since="1 month ago" | wc -l | xargs echo "Churn Detection:"
    git log --oneline --grep="(api)" --since="1 month ago" | wc -l | xargs echo "API:"
    git log --oneline --grep="(core)" --since="1 month ago" | wc -l | xargs echo "Core:"
    git log --oneline --grep="(customer)" --since="1 month ago" | wc -l | xargs echo "Customer:"
    
    # Recent commits
    echo -e "\n${BLUE}Recent Commits:${NC}"
    git log --oneline -10 --since="1 week ago"
}

# Main script logic
case $1 in
    "setup")
        setup_dev_env
        ;;
    "feature")
        create_feature_branch $2
        ;;
    "fix")
        create_fix_branch $2
        ;;
    "commit")
        smart_commit
        ;;
    "update")
        update_branch
        ;;
    "cleanup")
        cleanup_branches
        ;;
    "stats")
        show_milestone_stats
        ;;
    *)
        echo "Git Workflow Helper for Milestone 1"
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  setup              Setup development environment with pre-commit hooks"
        echo "  feature <name>     Create a new feature branch"
        echo "  fix <name>         Create a new fix branch"
        echo "  commit             Smart commit with change analysis"
        echo "  update             Update current branch with master changes"
        echo "  cleanup            Clean up merged branches"
        echo "  stats              Show Milestone 1 development statistics"
        echo ""
        echo "Examples:"
        echo "  $0 setup"
        echo "  $0 feature churn-detection-core"
        echo "  $0 fix customer-data-validation"
        echo "  $0 commit"
        ;;
esac