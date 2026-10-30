"""
Model settings and prompt-size budgets, in one place.

Ollama does NOT use a model's full context window by default -- it ran
llama3.1 (128k native) at 4096 tokens, and when a prompt is longer it
silently drops the *beginning*. Measured on this project: a 10.5k-token
prompt was cut to 2,050 evaluated tokens with no error. Every prompt Quill
builds is therefore sized against NUM_CTX below, and NUM_CTX is passed to
Ollama explicitly.
"""
import os

MODEL_NAME = os.environ.get("QUILL_MODEL", "llama3.1")
EMBEDDING_MODEL_NAME = os.environ.get("QUILL_EMBEDDING_MODEL", "nomic-embed-text")

# 16k tokens: llama3.1 8B (Q4) needs ~7 GB of VRAM at this size, which fits
# fully on a 12 GB GPU. Raise via QUILL_NUM_CTX if you have more memory.
NUM_CTX = int(os.environ.get("QUILL_NUM_CTX", "16384"))

# Conservative characters-per-token estimate for budgeting (measured ~3.2 on
# number-heavy text; English prose is closer to 4). Erring low means prompts
# are sized smaller than strictly necessary rather than overflowing.
CHARS_PER_TOKEN = 3.2

# Tokens set aside for the fixed instruction text wrapped around content.
PROMPT_OVERHEAD_TOKENS = 400


def input_char_budget(reserved_output_tokens: int, num_ctx: int = NUM_CTX) -> int:
    """How many characters of content a prompt can carry while leaving
    `reserved_output_tokens` of the context window for the model's reply
    (Ollama counts prompt + reply against the same window)."""
    usable = num_ctx - reserved_output_tokens - PROMPT_OVERHEAD_TOKENS
    if usable <= 0:
        raise ValueError(f"QUILL_NUM_CTX={num_ctx} is too small (need more than {reserved_output_tokens + PROMPT_OVERHEAD_TOKENS})")
    return int(usable * CHARS_PER_TOKEN)


# Replies to drafting/merge prompts are capped at this many tokens
# (num_predict), so the merge step's input -- several such replies -- has a
# known upper bound.
MAX_DRAFT_TOKENS = 2048

# Chat replies get this much room; history is trimmed to fit the rest.
CHAT_REPLY_TOKENS = 2048


def make_chat_model(**overrides):
    """The ChatOllama instance Quill uses everywhere. One shared instance
    with one num_ctx: a different num_ctx per call would make Ollama reload
    the model each time."""
    from langchain_ollama import ChatOllama

    params = {"model": MODEL_NAME, "temperature": 0, "num_ctx": NUM_CTX}
    params.update(overrides)
    return ChatOllama(**params)


def check_ollama(model) -> None:
    """Fails fast with a clear message if Ollama isn't reachable or the
    model isn't pulled -- constructing ChatOllama doesn't connect, so
    without this the first error only appears mid-conversation."""
    try:
        model.invoke("Reply with OK.")
    except Exception as e:
        raise RuntimeError(
            f"Could not reach Ollama model '{MODEL_NAME}': {e}. "
            f"Is Ollama running, and has the model been pulled (ollama pull {MODEL_NAME})?"
        ) from e


def trim_history(messages: list, max_chars: int) -> list:
    """
    The most recent messages whose combined text fits in `max_chars`
    (always at least the latest one). Sending the whole, ever-growing
    session history made long chats overflow the context window, and since
    Ollama drops a prompt's beginning, the first casualty was the system
    message carrying the retrieved document context.
    """
    kept, total = [], 0
    for message in reversed(messages):
        total += len(message.content)
        if kept and total > max_chars:
            break
        kept.append(message)
    return list(reversed(kept))
