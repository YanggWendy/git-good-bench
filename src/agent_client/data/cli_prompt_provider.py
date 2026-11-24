"""Prompts for a standard CLI coding agent.

The prompts of ``PromptProvider`` address the custom tools of the baseline
agent (``resolve_current_merge_conflict_with``, ``commit_changes_in``,
``update_rebase_todo_list``, ...). A CLI coding agent works in a shell inside
the scenario container instead, so this provider keeps the task overview, the
scenario context and the instructions of the original prompts verbatim and
replaces the tool sections with where the repository is and how to complete
the task with Git:

* merge: files with conflicts, all conflicts and the temporal ordering of the
  merge parents; conclude the merge with a commit.
* interactive rebase: the commits of the paused rebase; edit the todo list and
  continue. ``reword`` and ``squash`` are written the way the original
  ``execute_rebase`` executes them (``pick``/``fixup`` followed by ``exec git
  commit --amend -m``), because no editor is available.
* iterative committing: the hunks of ``file_changes.patch``; commit them in
  groups, then commit ``all_changes.patch`` (what ``commit_changes_in`` and
  ``commit_remaining_changes`` do).
"""
from typing import Dict

from src.agent_client.data.prompt_provider import PromptProvider
from src.agent_client.utils.available_context import AvailableContext


class CliPromptProvider:
    _USER_PROMPT_MERGE_CONFLICT = """{task_overview}
    The repository is checked out at `{repository_work_dir}` on the branch `current-scenario-branch`. The merge of the
    incoming changes (`MERGE_HEAD`) into this branch has already been started and stopped because of merge conflicts.
    Work directly in this checkout with the shell.

    The following files have merge conflicts:
    {files_with_conflicts}

    Below are all merge conflicts that need to be resolved, delimited by <CONFLICT-i> tags where i is the 0-based index:
    {all_merge_conflicts}

    Instructions:
    - Start with resolving the conflict at index 0 (CONFLICT-0) and proceed in ascending order through the conflicts.
    - Consider the context around the merge conflicts, of the overall diffs and files in which the conflicts occur.
    - Resolve the conflicts in a cohesive manner. For example, if you remove a function in a conflict, make sure that
        you also remove any invocations of that function in any other conflicts.
    - If you are just choosing one of the two sides, without changing any of the actual content, make sure to also reproduce
        the whitespaces exactly.
    - If the merge conflict occurs due to a NOP (e.g. one side of the conflict is empty, the other is a commented code block)
        favor resolving the conflict to the most maintainable and concise way. Avoid dead code.
    - Make sure to consider the implications your previous resolutions have on the remaining resolutions, especially when 
        resolving multiple conflicts in a single file.
    - If you find simple bugs, such as typos, copy and paste errors in variable assignments or parameters, feel free to 
        help your junior developer fix these. Do not perform complex refactorings or attempt to change code drastically. 
        Make as few changes to the side that you are accepting as possible.
    {instruction_details}

    Completing the task:
    - Once all conflicts are resolved, stage the resolved files and conclude the merge with a commit on
        `current-scenario-branch` (for example `git add <files> && git commit --no-edit`).
    - No interactive editor is available (`core.editor` is set to `true`), so pass commit messages on the command line.
    - When you are done, no merge may be in progress, no conflict markers may remain, and `current-scenario-branch`
        must point to your merge commit.

    Key Requirements:
    - Once the conflicts are resolved and the merge is committed you are done and must terminate.
    - Your decisions must be cohesive across merge conflicts.
    - Make sure that all your lines end with a newline character to avoid introducing accidental changes.
    - Do not make any changes to the repository beyond resolving the merge conflicts."""

    _USER_PROMPT_REBASE = """
    You are a staff software engineer with expertise in {programming_language} and Git.
    You are helping a junior team member who has been committing all day without pushing their commits to the remote. 
    Help them perform an interactive rebase to clean up their local tree. The rebase has already been initiated for you 
    and is currently paused so that you can inspect the commits participating in the rebase and edit the rebase todo list.

    The repository is checked out at `{repository_work_dir}`. The interactive rebase covers the last {times_seen_consecutively} commits of the
    branch `current-scenario-branch` and is paused before the first of them. Work directly in this checkout with the shell.

    The commits involved in the rebase are listed below in the order of the rebase todo list (`.git/rebase-merge/git-rebase-todo`),
    oldest first. Avoid viewing all commits again, they are already presented below. The commits are delimited by the <COMMIT-i> and </COMMIT-i> tags:
    {participating_commits}

    Instructions:
    Consider the changes in the commits and make adjustments if necessary such that the local tree:
    - contains logically cohesive commits
    - all commits have meaningful, descriptive commit messages that follow a unified format
    - does not contain commits with duplicate commit messages
    - follows best practices for maintainable code

    Completing the task:
    - Edit the rebase todo list, reordering its lines or changing their commands, and execute it with `git rebase --continue`.
        The lines are processed from top to bottom.
    - Only the following commands are allowed for the commits in the rebase todo list:
        - pick: Use this commit as is.
        - drop: Remove this commit.
        - fixup: Meld this commit into previous commit, reducing the total amount of commits by 1. Only keep the previous commit's log message.
        - fixup -C: Meld this commit into previous commit, reducing the total amount of commits by 1. Only keep this commit's log message.
        - squash: Meld this commit into previous commit, reducing the total amount of commits by 1. Commit message of resulting commit must be specified.
        - reword: Use commit, but edit commit message. Commit message must be specified.
    - No interactive editor is available (`core.editor` is set to `true`), so `squash` and `reword` cannot ask for the commit message.
        Specify it on the line after the commit instead: write `fixup` (for squash) or `pick` (for reword), followed by the line
        `exec git commit --amend -m "<commit message>"`.
    - When you are done, no rebase may be in progress and `current-scenario-branch` must point to the cleaned up history.

    Key Requirements:
    - You must not simply pick all commits without modifying anything in the rebase todo list. Do your best to improve the local tree however you see fit.
    - Avoid squashing all commits into a single commit, consider for which commits this would improve the resulting commit history.
    - Try to consolidate the total size of the local tree such that the resulting tree has length k<{times_seen_consecutively}
"""

    _USER_PROMPT_CHUNK = """You are a staff software engineer with expertise in {programming_language} and Git. You are helping a junior team member who has been working all day without creating a commit to iteratively create commits and introduce their changes into the repository in a maintainable way. Help them to select hunks such that you can create multiple, small, but logically cohesive commits that are structurally sound, and follow best practices for maintainable code.

    The repository is checked out at `{repository_work_dir}` on the branch `current-scenario-branch`. The changes of the junior team member are
    not applied to the working tree. They are stored in two patch files in the root of the repository, which Git does not track:
    - `file_changes.patch`: the changes to `{file}`. Its hunks are the hunks listed below, in the same order.
    - `all_changes.patch`: the changes to all other files. It is empty if there are none.
    Work directly in this checkout with the shell.

    Instructions:
    - Review the remaining hunks of code and help the junior engineer select the appropriate hunks for each commit.
    - Ensure that you select as many hunks as you need to ensure structural integrity, ie avoid breaking changes by, for example, removing a variable definition or initialization in one commit, but removing the usages of the variable in another commit.
    - Each commit should be focused, small, and logically cohesive.
    - Provide a clear and concise commit message for each commit following the format of the examples below.

    Key Requirements:
    - Avoid apply all changes in a single commit unless you are absolutely sure this will yield the best possible git history.

    Process all of the following {number_of_remaining_hunks} hunks:
    {remaining_hunks}

    Completing the task:
    - Create the commits on `current-scenario-branch`. For each commit, apply the hunks you selected to `{file}`, stage the file and
        commit it (for example `git apply --whitespace=fix <patch with the selected hunks>`, `git add {file}` and
        `git commit -m "<commit message>"`).
    - Once all hunks are committed, apply the remaining changes with `git apply --allow-empty --whitespace=fix all_changes.patch` and
        commit them in one final commit. Skip this step if `all_changes.patch` is empty.
    - Do not commit `file_changes.patch` and `all_changes.patch`.
    - No interactive editor is available (`core.editor` is set to `true`), so pass commit messages on the command line.
    - Examples of commit messages:
        "FIX: Handle edge case of uninitialized object"
        "ADD: Introduced new enum class CarConfiguration"
        "REFACTOR: Migrate car configurator to CarConfiguration enum"
        "UPDATE: Implement data streaming feature"
"""

    @classmethod
    def get_merge_prompt(cls, scenario: Dict, context: Dict, repository_work_dir: str) -> str:
        """
        Args:
            scenario: The merge scenario (parsed ``SampleDataRowV4.scenario``).
            context: Scenario context as returned by ``ScenarioEnvironmentManager.provide_scenario_context`` for the
                merge contexts, plus ``AvailableContext.PROGRAMMING_LANGUAGE``.
            repository_work_dir: Path of the repository inside the container.

        Returns:
            str: The user prompt for the CLI agent.
        """
        task_overview = PromptProvider._MERGE_PROMPT_TASK_OVERVIEW.format(
            commit_temporal_ordering=context[AvailableContext.COMMIT_TEMPORAL_ORDERING],
            programming_language=context[AvailableContext.PROGRAMMING_LANGUAGE],
            total_amount_of_merge_conflicts=context[AvailableContext.TOTAL_AMOUNT_OF_MERGE_CONFLICTS],
        )
        return cls._USER_PROMPT_MERGE_CONFLICT.format(
            task_overview=task_overview,
            repository_work_dir=repository_work_dir,
            files_with_conflicts=context[AvailableContext.FILES_WITH_CONFLICTS],
            all_merge_conflicts=context[AvailableContext.ALL_MERGE_CONFLICTS],
            instruction_details=PromptProvider._MERGE_PROMPT_INSTRUCTION_DETAILS,
        )

    @classmethod
    def get_rebase_prompt(cls, scenario: Dict, context: Dict, repository_work_dir: str) -> str:
        """
        Args:
            scenario: The file-commit-chain scenario (parsed ``SampleDataRowV4.scenario``).
            context: ``AvailableContext.REBASE_PARTICIPATING_COMMITS`` and ``AvailableContext.PROGRAMMING_LANGUAGE``.
            repository_work_dir: Path of the repository inside the container.

        Returns:
            str: The user prompt for the CLI agent.
        """
        return cls._USER_PROMPT_REBASE.format(
            programming_language=context[AvailableContext.PROGRAMMING_LANGUAGE],
            repository_work_dir=repository_work_dir,
            times_seen_consecutively=scenario['times_seen_consecutively'],
            participating_commits=context[AvailableContext.REBASE_PARTICIPATING_COMMITS],
        )

    @classmethod
    def get_chunk_prompt(cls, scenario: Dict, context: Dict, repository_work_dir: str) -> str:
        """
        Args:
            scenario: The file-commit-chain scenario (parsed ``SampleDataRowV4.scenario``).
            context: ``AvailableContext.REMAINING_HUNKS`` (number of hunks, hunks) and
                ``AvailableContext.PROGRAMMING_LANGUAGE``.
            repository_work_dir: Path of the repository inside the container.

        Returns:
            str: The user prompt for the CLI agent.
        """
        return cls._USER_PROMPT_CHUNK.format(
            programming_language=context[AvailableContext.PROGRAMMING_LANGUAGE],
            repository_work_dir=repository_work_dir,
            file=scenario['file'],
            number_of_remaining_hunks=context[AvailableContext.REMAINING_HUNKS][0],
            remaining_hunks=context[AvailableContext.REMAINING_HUNKS][1],
        )
