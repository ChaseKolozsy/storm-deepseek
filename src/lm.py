import logging
import os
import random
import threading
from typing import Optional, Literal, Any

import backoff
import dspy
import requests
from dsp import ERRORS, backoff_hdlr, giveup_hdlr
from dsp.modules.hf import openai_to_hf
from dsp.modules.hf_client import send_hfvllm_request_v00, send_hftgi_request_v01_wrapped
from transformers import AutoTokenizer
from openai import OpenAI

try:
    from anthropic import RateLimitError
except ImportError:
    RateLimitError = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s')

class OpenAIModel(dspy.OpenAI):
    """A wrapper class for dspy.OpenAI."""

    def __init__(
            self,
            model: str = "deepseek-chat",
            api_key: Optional[str] = None,
            api_provider: Literal["openai", "azure", "deepseek"] = "openai",
            api_base: Optional[str] = None,
            model_type: Literal["chat", "text"] = None,
            **kwargs
    ):
        self.api_provider = api_provider
        self.api_base = api_base
        self.model = model
        self.api_key = api_key

        if api_provider == "deepseek":
            self.api_base = self.api_base or "https://api.deepseek.com/v1"
            model_type = "chat"  # Assuming deepseek-chat is always a chat model
            self.model = "deepseek-chat"  # Ensure we're using the correct model name

        # Create a custom OpenAI client with the specified api_base
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)

        super().__init__(model=self.model, api_key=self.api_key, api_base=self.api_base,
                         model_type=model_type, **kwargs)
        self._token_usage_lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def log_usage(self, response):
        """Log the total tokens from the OpenAI API response."""
        usage_data = response.get('usage')
        if usage_data:
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.get('prompt_tokens', 0)
                self.completion_tokens += usage_data.get('completion_tokens', 0)

    def get_usage_and_reset(self):
        """Get the total tokens used and reset the token usage."""
        usage = {
            self.model: {'prompt_tokens': self.prompt_tokens, 'completion_tokens': self.completion_tokens}
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0

        return usage

    def __call__(
            self,
            prompt: str,
            only_completed: bool = True,
            return_sorted: bool = False,
            **kwargs,
    ) -> list[dict[str, Any]]:
        """Copied from dspy/dsp/modules/gpt3.py with the addition of tracking token usage."""

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        response = self.request(prompt, **kwargs)
        self.log_usage(response)

        choices = response["choices"]
        completed_choices = [c for c in choices if c["finish_reason"] != "length"]

        if only_completed and len(completed_choices):
            choices = completed_choices

        completions = [self._get_choice_text(c) for c in choices]

        return completions

    def request(self, prompt, **kwargs):
        """Override the request method to handle deepseek-chat specifics if needed."""
        logging.debug(f"OpenAIModel.request received kwargs: {kwargs}")
        
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}")
        
        allowed_kwargs = {'temperature', 'top_p', 'n', 'stream', 'stop', 'max_tokens',
                          'presence_penalty', 'frequency_penalty', 'logit_bias', 'user'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_kwargs}
        
        # Ensure we're always passing the required 'model' and 'messages' parameters
        filtered_kwargs['model'] = self.model
        filtered_kwargs['messages'] = messages
        
        try:
            # Use the custom client to make the API call
            response = self.client.chat.completions.create(**filtered_kwargs)
            return response.model_dump()
        except Exception as e:
            logging.error(f"Error in API call: {str(e)}")
            raise

    def _get_choice_text(self, choice):
        """Override to handle different response formats."""
        if self.api_provider == "deepseek":
            return choice.get("message", {}).get("content", "")
        return super()._get_choice_text(choice)