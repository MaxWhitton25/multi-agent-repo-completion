import aider
import base64
import hashlib
import json
import locale
import math
import mimetypes
import os
import platform
import re
import sys
import threading
import time
import traceback
import webbrowser
import re
from collections import defaultdict
from datetime import datetime
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import List

from aider import __version__, models, prompts, urls, utils
from aider.analytics import Analytics
from aider.commands import Commands
from aider.history import ChatSummary
from aider.io import ConfirmGroup, InputOutput
from aider.linter import Linter
from aider.llm import litellm
from aider.repo import ANY_GIT_ERROR, GitRepo
from aider.repomap import RepoMap
from aider.run_cmd import run_cmd
from aider.sendchat import RETRY_TIMEOUT, retry_exceptions, send_completion
from aider.utils import format_content, format_messages, format_tokens, is_image_file

from ..dump import dump  # noqa: F401
from .chat_chunks import ChatChunks
class Rollbback(aider.Coder):
        def send_message(self, inp):
            import openai  # for error codes below

            self.cur_messages += [
                dict(role="user", content=inp),
            ]

            chunks = self.format_messages()
            messages = chunks.all_messages()
            self.warm_cache(chunks)

            if self.verbose:
                utils.show_messages(messages, functions=self.functions)

            self.multi_response_content = ""
            if self.show_pretty() and self.stream:
                self.mdstream = self.io.get_assistant_mdstream()
            else:
                self.mdstream = None

            retry_delay = 0.125

            self.usage_report = None
            exhausted = False
            interrupted = False
            self.io.append_chat_history(f"We are on reflection number: {self.num_reflections} of {self.max_reflections}")

            try:
                while True:
                    try:
                        yield from self.send(messages, functions=self.functions)
                        break
                    except retry_exceptions() as err:
                        # Print the error and its base classes
                        # for cls in err.__class__.__mro__: dump(cls.__name__)

                        retry_delay *= 2
                        if retry_delay > RETRY_TIMEOUT:
                            self.mdstream = None
                            self.check_and_open_urls(err)
                            break
                        err_msg = str(err)
                        self.io.tool_error(err_msg)
                        self.io.tool_output(f"Retrying in {retry_delay:.1f} seconds...")
                        time.sleep(retry_delay)
                        continue
                    except KeyboardInterrupt:
                        interrupted = True
                        break
                    except litellm.ContextWindowExceededError:
                        # The input is overflowing the context window!
                        exhausted = True
                        break
                    except litellm.exceptions.BadRequestError as br_err:
                        self.io.tool_error(f"BadRequestError: {br_err}")
                        return
                    except FinishReasonLength:
                        # We hit the output limit!
                        if not self.main_model.info.get("supports_assistant_prefill"):
                            exhausted = True
                            break

                        self.multi_response_content = self.get_multi_response_content()

                        if messages[-1]["role"] == "assistant":
                            messages[-1]["content"] = self.multi_response_content
                        else:
                            messages.append(
                                dict(role="assistant", content=self.multi_response_content, prefix=True)
                            )
                    except (openai.APIError, openai.APIStatusError) as err:
                        # for cls in err.__class__.__mro__: dump(cls.__name__)
                        self.mdstream = None
                        self.check_and_open_urls(err)
                        break
                    except Exception as err:
                        lines = traceback.format_exception(type(err), err, err.__traceback__)
                        self.io.tool_warning("".join(lines))
                        self.io.tool_error(str(err))
                        return
            finally:
                if self.mdstream:
                    self.live_incremental_response(True)
                    self.mdstream = None

                self.partial_response_content = self.get_multi_response_content(True)
                self.multi_response_content = ""

            self.io.tool_output()

            self.show_usage_report()

            if exhausted:
                self.show_exhausted_error()
                self.num_exhausted_context_windows += 1
                return

            if self.partial_response_function_call:
                args = self.parse_partial_args()
                if args:
                    content = args.get("explanation") or ""
                else:
                    content = ""
            elif self.partial_response_content:
                content = self.partial_response_content
            else:
                content = ""

            try:
                self.reply_completed()
            except KeyboardInterrupt:
                interrupted = True

            if interrupted:
                content += "\n^C KeyboardInterrupt"
                self.cur_messages += [dict(role="assistant", content=content)]
                return

            edited = self.apply_updates()

            self.update_cur_messages()

            if edited:
                self.aider_edited_files.update(edited)
                saved_message = self.auto_commit(edited)

                if not saved_message and hasattr(self.gpt_prompts, "files_content_gpt_edits_no_repo"):
                    saved_message = self.gpt_prompts.files_content_gpt_edits_no_repo

                self.move_back_cur_messages(saved_message)

            if self.reflected_message:
                return

            if edited and self.auto_lint:
                lint_errors = self.lint_edited(edited)
                self.auto_commit(edited, context="Ran the linter")
                self.lint_outcome = not lint_errors
                if lint_errors:
                    ok = self.io.confirm_ask("Attempt to fix lint errors?")
                    if ok:
                        self.reflected_message = lint_errors
                        self.update_cur_messages()
                        return

            shared_output = self.run_shell_commands()
            if shared_output:
                self.cur_messages += [
                    dict(role="user", content=shared_output),
                    dict(role="assistant", content="Ok"),
                ]

            if edited and self.auto_test:
                test_errors = self.commands.cmd_test(self.test_cmd)
                self.test_outcome = not test_errors
                passed, failed = parse_test_results(test_errors)

                if test_errors:
                    if performance_improving(self.test_history, passed, failed):
                        self.test_history.append((passed, failed))
                        ok = self.io.confirm_ask("Attempt to fix test errors?")
                        if ok:
                            self.reflected_message = test_errors
                            self.update_cur_messages()
                            return
                    else:
                        self.io.append_chat_history("The performance did not improve, so we're going to roll back.")
                        self.num_reflections+=1
                        self.test_history = []
                        ok = self.io.confirm_ask("Attempt to roll back?")
                        if ok:
                            self.repo.revert_to_commit(self.repo.initial_commit_hash)
                            self.reflected_message = test_errors
                            self.update_cur_messages()
                            return


            add_rel_files_message = self.check_for_file_mentions(content)
            if add_rel_files_message:
                if self.reflected_message:
                    self.reflected_message += "\n\n" + add_rel_files_message
                else:
                    self.reflected_message = add_rel_files_message
        def parse_test_results(test_string):
            failed = -1
            passed = -1  
            try:
                failed_match = re.search(r'(\d+)\s+failed', test_string)
                if failed_match:
                    failed = int(failed_match.group(1))
                passed_match = re.search(r'(\d+)\s+passed', test_string)
                if passed_match:
                    passed = int(passed_match.group(1))       
                return (failed, passed)
            except Exception:
                return (-1, -1)
        def performance_improving(history, passed, failed):
            if history == []:
                return True
            if history[-1][0] + history[-1][1] != passed+failed:
                return True
            return passed<history[-1][0]