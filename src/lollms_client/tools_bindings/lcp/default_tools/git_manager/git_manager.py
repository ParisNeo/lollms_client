import os
from typing import Dict, Any, Optional, List
from pathlib import Path

try:
    import git
    from git.exc import GitCommandError, InvalidGitRepositoryError, NoSuchPathError
except ImportError:
    git = None
    GitCommandError = None
    InvalidGitRepositoryError = None
    NoSuchPathError = None

TOOL_LIBRARY_NAME = "Git Manager"
TOOL_LIBRARY_DESC = "Provides structured, safe git operations (status, add, commit, branch, diff) without shell execution."
TOOL_LIBRARY_ICON = "🌿"

def init_tools_library(config: dict = None) -> None:
    global git, GitCommandError, InvalidGitRepositoryError, NoSuchPathError
    if git is None:
        try:
            import pipmaster as pm
            pm.ensure_packages("gitpython")
            import git as _git
            git = _git
            from git import exc as _git_exc
            GitCommandError = _git_exc.GitCommandError
            InvalidGitRepositoryError = _git_exc.InvalidGitRepositoryError
            NoSuchPathError = _git_exc.NoSuchPathError
        except Exception as e:
            import ascii_colors
            ascii_colors.ASCIIColors.error(f"[Git Manager] Failed to install/import gitpython: {e}")
    elif GitCommandError is None:
        try:
            from git import exc as _git_exc
            GitCommandError = _git_exc.GitCommandError
            InvalidGitRepositoryError = _git_exc.InvalidGitRepositoryError
            NoSuchPathError = _git_exc.NoSuchPathError
        except Exception:
            pass

def _get_repo() -> Optional[Any]:
    if git is None:
        return None
    try:
        repo_path = Path.cwd()
        return git.Repo(repo_path, search_parent_directories=True)
    except Exception:
        return None

def tool_git_status() -> Dict[str, Any]:
    """
    Returns the current git status of the workspace, including staged, unstaged, and untracked files.
    """
    if git is None:
        return {"success": False, "error": "GitPython is not installed. Cannot perform git operations."}
    
    repo = _get_repo()
    if not repo:
        return {"success": False, "error": "Not a git repository (or any of the parent directories)."}
    
    try:
        staged = [item.a_path for item in repo.index.diff("HEAD")]
        unstaged = [item.a_path for item in repo.index.diff(None)]
        untracked = repo.untracked_files
        
        is_dirty = repo.is_dirty()
        active_branch = repo.active_branch.name if not repo.head.is_detached else "detached"
        
        status_str = f"On branch {active_branch}\n"
        if is_dirty:
            status_str += "Changes detected:\n"
            if staged:
                status_str += "Staged files:\n"
                for f in staged:
                    status_str += f"  modified:   {f}\n"
            if unstaged:
                status_str += "Changes not staged for commit:\n"
                for f in unstaged:
                    status_str += f"  modified:   {f}\n"
            if untracked:
                status_str += "Untracked files:\n"
                for f in untracked:
                    status_str += f"  {f}\n"
        else:
            status_str += "nothing to commit, working tree clean."
            
        return {
            "success": True,
            "output": status_str,
            "is_dirty": is_dirty,
            "active_branch": active_branch,
            "staged": staged,
            "unstaged": unstaged,
            "untracked": untracked
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to get git status: {str(e)}"}

def tool_git_diff(staged: bool = False) -> Dict[str, Any]:
    """
    Returns the git diff of the workspace.
    
    Args:
        staged (bool, optional): If True, returns the diff of staged changes. Otherwise, returns unstaged changes. Defaults to False.
    """
    if git is None:
        return {"success": False, "error": "GitPython is not installed."}
        
    repo = _get_repo()
    if not repo:
        return {"success": False, "error": "Not a git repository."}
        
    try:
        if staged:
            diff = repo.git.diff("--cached")
        else:
            diff = repo.git.diff()
            
        if not diff.strip():
            return {"success": True, "output": "No changes detected."}
            
        return {
            "success": True,
            "output": diff
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to get git diff: {str(e)}"}

def tool_git_commit(message: str, add_all: bool = True) -> Dict[str, Any]:
    """
    Stages and commits changes to the repository.
    
    Args:
        message (str): The commit message.
        add_all (bool, optional): If True, stages all modified and untracked files before committing. Defaults to True.
    """
    if git is None:
        return {"success": False, "error": "GitPython is not installed."}
        
    repo = _get_repo()
    if not repo:
        return {"success": False, "error": "Not a git repository."}
        
    try:
        if add_all:
            repo.git.add(A=True)
            
        if not repo.is_dirty() and not repo.untracked_files:
            return {"success": True, "output": "Nothing to commit, working tree clean."}
            
        repo.index.commit(message)
        return {
            "success": True,
            "output": f"Successfully committed changes with message: '{message}'"
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to commit: {str(e)}"}

def tool_git_create_branch(branch_name: str, checkout: bool = True) -> Dict[str, Any]:
    """
    Creates a new git branch.
    
    Args:
        branch_name (str): The name of the new branch.
        checkout (bool, optional): If True, checks out the new branch immediately. Defaults to True.
    """
    if git is None:
        return {"success": False, "error": "GitPython is not installed."}
        
    repo = _get_repo()
    if not repo:
        return {"success": False, "error": "Not a git repository."}
        
    try:
        if branch_name in [b.name for b in repo.branches]:
            return {"success": False, "error": f"Branch '{branch_name}' already exists."}
            
        new_branch = repo.create_head(branch_name)
        if checkout:
            new_branch.checkout()
            
        return {
            "success": True,
            "output": f"Branch '{branch_name}' created" + (" and checked out." if checkout else ".")
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to create branch: {str(e)}"}

def tool_git_checkout(branch_name: str) -> Dict[str, Any]:
    """
    Checks out an existing git branch.
    
    Args:
        branch_name (str): The name of the branch to check out.
    """
    if git is None:
        return {"success": False, "error": "GitPython is not installed."}
        
    repo = _get_repo()
    if not repo:
        return {"success": False, "error": "Not a git repository."}
        
    try:
        if repo.is_dirty():
            return {
                "success": False,
                "error": "Cannot checkout branch: working tree is dirty. Please commit or stash changes first."
            }
            
        if branch_name not in [b.name for b in repo.branches]:
            return {"success": False, "error": f"Branch '{branch_name}' does not exist."}
            
        repo.git.checkout(branch_name)
        return {
            "success": True,
            "output": f"Switched to branch '{branch_name}'."
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to checkout branch: {str(e)}"}

def tool_git_log(max_count: int = 10) -> Dict[str, Any]:
    """
    Returns the git commit history.
    
    Args:
        max_count (int, optional): Maximum number of commits to return. Defaults to 10.
    """
    if git is None:
        return {"success": False, "error": "GitPython is not installed."}
        
    repo = _get_repo()
    if not repo:
        return {"success": False, "error": "Not a git repository."}
        
    try:
        commits = list(repo.iter_commits(max_count=max_count))
        log_str = ""
        for commit in commits:
            log_str += f"commit {commit.hexsha}\nAuthor: {commit.author.name} <{commit.author.email}>\nDate:   {commit.committed_datetime}\n\n    {commit.message}\n\n"
            
        return {
            "success": True,
            "output": log_str if log_str else "No commits yet."
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to get git log: {str(e)}"}

def tool_git_config_get(key: str, scope: str = "local") -> Dict[str, Any]:
    """
    Retrieves a git configuration value.

    Args:
        key (str): The git config key (e.g., 'user.name', 'user.email').
        scope (str, optional): The configuration scope. Options: 'local' (this repo only) or 'global' (user-wide). Defaults to 'local'.
    """
    if git is None:
        return {"success": False, "error": "GitPython is not installed. Run 'pip install gitpython' to enable git operations."}

    if scope not in ("local", "global"):
        return {"success": False, "error": f"Invalid scope '{scope}'. Must be 'local' or 'global'."}

    value = None
    try:
        if scope == "local":
            repo = _get_repo()
            if not repo:
                return {"success": False, "error": "Not a git repository. Use scope='global' to read global config."}
            value = repo.git.config("--get", "--local", key)
        else:
            value = git.cmd.Git().config("--get", "--global", key)
    except Exception as e:
        raw_err = str(e).strip()
        if not raw_err or raw_err.isdigit() or "exit code" in raw_err.lower() or "status code" in raw_err.lower() or "returned" in raw_err.lower() or "code 1" in raw_err.lower():
            return {"success": False, "error": f"Configuration key '{key}' is not set in {scope} scope. (Git exited with an error code, which typically means the key does not exist.)"}
        return {"success": False, "error": f"Failed to get git config for key '{key}' in {scope} scope: {raw_err}"}

    if value is None or str(value).strip() == "":
        return {"success": False, "error": f"Configuration key '{key}' is not set in {scope} scope."}

    return {
        "success": True,
        "output": f"{key} = {str(value).strip()}"
    }

def tool_git_config_set(key: str, value: str, scope: str = "local") -> Dict[str, Any]:
    """
    Sets a git configuration value.

    Args:
        key (str): The git config key (e.g., 'user.name').
        value (str): The value to set.
        scope (str, optional): The configuration scope. Options: 'local' (this repo only) or 'global' (user-wide). Defaults to 'local'.
    """
    if git is None:
        return {"success": False, "error": "GitPython is not installed."}

    if scope not in ("local", "global"):
        return {"success": False, "error": "Invalid scope. Must be 'local' or 'global'."}

    try:
        if scope == "local":
            repo = _get_repo()
            if not repo:
                return {"success": False, "error": "Not a git repository. Use scope='global' to write global config."}
            repo.git.config("--local", key, value)
        else:
            git.cmd.Git().config("--global", key, value)

        return {
            "success": True,
            "output": f"Successfully set {key} to {value} in {scope} scope."
        }
    except Exception as e:
        raw_err = str(e).strip()
        if not raw_err:
            raw_err = f"git config set failed for key '{key}' with an empty error message."
        return {"success": False, "error": f"Failed to set git config for key '{key}' in {scope} scope: {raw_err}"}