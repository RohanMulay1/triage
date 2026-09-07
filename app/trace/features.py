"""The served and fan-out paths share the same prompt feature definitions."""
from ..router import prefilter
from ..router.longcontext import long_context_route


def root_prompt_features(question: str, long_context: dict) -> dict:
    return {"long_context": long_context["long_context"],
            "approx_prompt_tokens": long_context["approx_prompt_tokens"],
            "message_chars": len(question)}


def prefilter_prompt_features(prediction: dict, memory: dict) -> dict:
    return {"prefilter_route": prediction["route"],
            "predicted_difficulty": prediction["difficulty"],
            "memory_hit": memory.get("hit"), "memory_neighbours": memory.get("n")}


def build_prompt_features(question: str, system: str, cfg: dict) -> dict:
    """A frozen cold-memory control; no held-out outcome enters prompt features."""
    lc = long_context_route(system + " " + question, cfg["tiers"]["long_context_tokens"])
    mem = {"hit": False, "n": 0}
    pcfg = cfg.get("prefilter", {})
    pf = (prefilter.predict(question, mem, pcfg) if pcfg.get("enabled", True)
          else {"route": "normal", "difficulty": None})
    return {**root_prompt_features(question, lc), **prefilter_prompt_features(pf, mem),
            "prompt_feature_context": "cold_memory"}
