"""
Central LLM configuration.

CrewAI Agents need an `llm` object passed in explicitly — otherwise CrewAI
falls back to OpenAI's gpt-4 by default and requires OPENAI_API_KEY, which is
what was causing the failures in this project.

This module builds a single shared LLM (Groq, free tier) and every agent in
agents.py imports `llm` from here instead of each agent picking its own
default.

Setup:
    1. Get a free API key from https://console.groq.com/keys
    2. Put it in a `.env` file in this folder:
           GROQ_API_KEY=your-key-here
    3. (Optional) override the model via a GROQ_MODEL env var, e.g.:
           GROQ_MODEL=llama-3.3-70b-versatile
"""

import os

from crewai import LLM
from dotenv import load_dotenv

load_dotenv()

# --- Workaround for crewAI issue #5886 -------------------------------------
# CrewAI tags every message with an internal "cache_breakpoint" flag used for
# Anthropic's prompt-caching feature. It's supposed to strip that flag before
# sending to non-Anthropic providers, but as of crewai 1.15.17 it doesn't for
# Groq (and other LiteLLM-routed providers) — so Groq's API rejects the
# request with:
#   GroqException - 'messages.0': property 'cache_breakpoint' is unsupported
# This is a confirmed open bug: https://github.com/crewAIInc/crewAI/issues/5886
# The fix below is the workaround posted by the crewAI maintainers in that
# thread. Safe to remove once crewai ships PR #5914 (or later) in a release.
import crewai.llms.cache as _crewai_cache

_crewai_cache.mark_cache_breakpoint = lambda msg: msg
# -----------------------------------------------------------------------------

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


def get_llm() -> LLM:
    """Return the shared LLM instance used by all agents."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set.\n"
            "1. Get a free key at https://console.groq.com/keys\n"
            "2. Create a .env file in this folder with:\n"
            "       GROQ_API_KEY=your-key-here\n"
        )

    model_name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

    # --- Automatic fallback when Groq hits its rate limit ------------------
    # litellm has a native fallback list: if the primary model's request
    # fails (e.g. Groq's RateLimitError), it automatically retries the same
    # request against the next model in the list, using its own API key.
    # Gemini's free tier (Google AI Studio) is used here since it's a
    # completely separate quota from Groq's — a Groq rate limit has no
    # effect on it. This only activates if GEMINI_API_KEY is set; without
    # it, the crew just runs on Groq with retries as before.
    fallbacks = []
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        fallbacks.append({"model": "gemini/gemini-3.6-flash", "api_key": gemini_key})
    # -------------------------------------------------------------------------

    return LLM(
        model=f"groq/{model_name}",
        api_key=api_key,
        temperature=0.7,
        max_retries=5,
        max_tokens=3000,
        fallbacks=fallbacks,
    )


llm = None
try:
    llm = get_llm()
except EnvironmentError:
    pass