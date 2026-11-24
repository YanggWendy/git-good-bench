"""LLM client for the LLM-as-a-judge evaluation of the file-commit-chain scenarios.

The released ``Evaluator`` builds its prompts with a ``Chat`` object and sends
them through an ``llm_client``; neither is part of the release. ``Chat`` holds
the messages and ``JudgeClient`` sends them to an OpenAI-compatible chat
completions API with ``judge.py``, the module the Harbor adapter uses in its
verifier (the two files are identical).
"""
from src.agent_client.utils import judge


class Chat:
    """The prompt builder the Evaluator uses: ``Chat().add_system(...).add_user(...)``."""

    def __init__(self):
        self.messages = []

    def add_system(self, content: str) -> 'Chat':
        self.messages.append({'role': 'system', 'content': content})
        return self

    def add_user(self, content: str) -> 'Chat':
        self.messages.append({'role': 'user', 'content': content})
        return self


class JudgeResponse:
    def __init__(self, content: str):
        self.content = content


class JudgeClient:
    """Answers the Evaluator's prompts with the verdict of the LLM judge.

    The Evaluator reads ``response.content`` with ``ast.literal_eval``, so the
    verdict is returned as the text of a Python dictionary. Temperature 0 and
    the response schema, which the Evaluator passes, are what ``judge.chat``
    always sends.
    """

    def chat(self, prompt: Chat, temperature: float = 0, response_schema: dict = None) -> JudgeResponse:
        system = [message['content'] for message in prompt.messages if message['role'] == 'system']
        user = [message['content'] for message in prompt.messages if message['role'] == 'user']
        if system != [judge.SYSTEM_PROMPT] or len(user) != 1 or temperature != 0:
            raise ValueError('Unexpected judge prompt')
        return JudgeResponse(repr(judge.verdict_for(user[0])))
