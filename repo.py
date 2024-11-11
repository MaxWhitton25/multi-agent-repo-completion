import os
import time
from pathlib import Path, PurePosixPath

import git
import pathspec
import aider

from aider import prompts, utils
from aider.sendchat import simple_send_with_retries

from .dump import dump  # noqa: F401

ANY_GIT_ERROR = (
    git.exc.ODBError,
    git.exc.GitError,
    OSError,
    IndexError,
    BufferError,
    TypeError,
    ValueError,
)

class RevertableRepo(aider.GitRepo):
    def __init__(self, *args, **kwargs):
        super.__init__(*args, **kwargs)
        # used for reverting
        self.initial_commit_hash = self.get_head_commit_sha()  
        self.previous_commit_hashes = [] 
    def revert_to_commit(self, commit_hash):
        """
        Check out the specified commit hash, reverting the repository to that state.

        Parameters:
            commit_hash (str): The commit hash to revert to.
        """
        current_commit_hash = self.get_head_commit_sha()

        if current_commit_hash and current_commit_hash not in self.previous_commit_hashes:
            self.previous_commit_hashes.append(current_commit_hash)

        try:
            self.repo.git.checkout(commit_hash)
            self.io.tool_output(f"Reverted to commit {commit_hash}.", bold=True)
        except git.exc.GitError as err:
            self.io.tool_error(f"Unable to checkout commit {commit_hash}: {err}")
    