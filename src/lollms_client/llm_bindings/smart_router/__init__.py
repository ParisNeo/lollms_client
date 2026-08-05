import re
import math
from typing import Dict, List, Optional, Any, Union, Callable
from collections import Counter
from ascii_colors import ASCIIColors
from lollms_client.lollms_llm_binding import LollmsLLMBinding, LollmsLLMBindingManager
from lollms_client.lollms_types import MSG_TYPE

class LollmsSmartBinding(LollmsLLMBinding):
    """
    A proxy binding that routes generation to child bindings based on routing profiles.
    It evaluates prompts using TF-IDF (subject), heuristics (complexity), and weights (cost/latency).
    """
    
    def __init__(self, 
                 binding_name: str = "SmartRouter",
                 model_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
                 routing_strategy: str = "balanced",
                 **kwargs):
        super().__init__(binding_name=binding_name, **kwargs)
        self.model_profiles = model_profiles or {}
        self.routing_strategy = routing_strategy
        
        # Mapping of alias -> actual binding instance
        self.child_bindings: Dict[str, LollmsLLMBinding] = {}
        
        # Instantiate child bindings lazily via the manager
        self._binding_manager = LollmsLLMBindingManager()
        
        # Default weights for "balanced" strategy
        self.weights = self._get_strategy_weights(routing_strategy)
        
        self._init_child_bindings()
        
    def _get_strategy_weights(self, strategy: str) -> Dict[str, float]:
        """Returns normalized weights based on the chosen strategy."""
        if strategy == "cost_optimized":
            return {"subject": 0.2, "complexity": 0.2, "cost": 0.5, "latency": 0.1}
        elif strategy == "quality_optimized":
            return {"subject": 0.4, "complexity": 0.4, "cost": 0.0, "latency": 0.2}
        else: # balanced
            return {"subject": 0.35, "complexity": 0.35, "cost": 0.15, "latency": 0.15}

    def _init_child_bindings(self):
        """Instantiates the child bindings defined in the profiles."""
        for alias, profile in self.model_profiles.items():
            b_name = profile.get("binding_name")
            b_config = profile.get("binding_config", {}) or {}
            
            # Inject client-level config into child
            b_config['user_name'] = self.user_name
            b_config['ai_name'] = self.ai_name
            b_config['debug'] = self.debug
            
            try:
                binding = self._binding_manager.create_binding(
                    binding_name=b_name,
                    **{k: v for k, v in b_config.items() if k != "binding_name"}
                )
                if binding:
                    binding.vision_enabled = profile.get("vision_enabled", False)
                    binding.forced_context_size = profile.get("forced_context_size")
                    binding.routing_profile = profile.get("routing_profile", {})
                    self.child_bindings[alias] = binding
                    ASCIIColors.info(f"[SmartRouter] Initialized child binding: {alias} ({b_name})")
                else:
                    ASCIIColors.error(f"[SmartRouter] Failed to create child binding: {alias}")
            except Exception as e:
                ASCIIColors.error(f"[SmartRouter] Error initializing child {alias}: {e}")

    def _calculate_tfidf_similarity(self, prompt: str, doc: str) -> float:
        def stem(word):
            if len(word) > 4:
                if word.endswith('ing'): return word[:-3]
                if word.endswith('ed'): return word[:-2]
                if word.endswith('s'): return word[:-1]
            return word

        def tokenize(text): 
            return [stem(w) for w in re.findall(r'\b\w+\b', text.lower())]

        prompt_tokens, doc_tokens = tokenize(prompt), tokenize(doc)
        if not prompt_tokens or not doc_tokens: return 0.0

        prompt_counts, doc_counts = Counter(prompt_tokens), Counter(doc_tokens)
        prompt_len, doc_len = len(prompt_tokens), len(doc_tokens)

        unique_tokens = set(prompt_tokens + doc_tokens)
        # Epsilon added to prevent ZeroDivisionError if token appears in both
        idf = {t: math.log(3 / (sum([1 for d in [prompt_counts, doc_counts] if t in d]) + 1 + 1e-9)) + 1 for t in unique_tokens}

        p_vec = {t: (c/prompt_len) * idf[t] for t, c in prompt_counts.items()}
        d_vec = {t: (c/doc_len) * idf[t] for t, c in doc_counts.items()}

        intersection = set(p_vec.keys()) & set(d_vec.keys())
        numerator = sum([p_vec[t] * d_vec[t] for t in intersection])

        sum_sq_p = sum([v**2 for v in p_vec.values()])
        sum_sq_d = sum([v**2 for v in d_vec.values()])

        # Epsilon added to denominator to prevent ZeroDivisionError on zero vectors
        denominator = math.sqrt(sum_sq_p) * math.sqrt(sum_sq_d) + 1e-9
        return numerator / denominator

    def _evaluate_complexity(self, prompt: str) -> int:
        prompt_lower = prompt.lower()
        length = len(prompt.split())
        if any(t in prompt_lower for t in ["prove", "architect", "refactor", "optimize", "derive"]) or length > 500: return 3
        if any(t in prompt_lower for t in ["write a script", "explain how", "summarize", "compare"]) or length > 100: return 2
        return 1

    def _select_model(self, prompt: str, images: Optional[List[str]] = None) -> Optional[str]:
        best_alias, best_score = None, -float('inf')
        complexity = self._evaluate_complexity(prompt)

        costs = [m.get("routing_profile", {}).get("cost_per_1k_tokens", 0.0) for m in self.model_profiles.values()]
        max_cost = max(costs) if costs else 0.0

        for alias, binding in self.child_bindings.items():
            profile = self.model_profiles[alias].get("routing_profile", {})

            # Hard filter: Vision
            if images and not getattr(binding, "vision_enabled", False):
                continue

            subject_match = self._calculate_tfidf_similarity(prompt, profile.get("description", ""))
            complexity_match = 1.0 - abs(complexity - profile.get("complexity_tier", 1)) / 3.0

            # Epsilon guard for cost normalization
            normalized_cost = profile.get("cost_per_1k_tokens", 0.0) / (max_cost + 1e-9) if max_cost > 0 else 0.0
            latency_penalty = -min(profile.get("avg_latency_ms", 100) / 2000.0, 1.0)

            w = self.weights
            score = (w.get("subject", 0.3) * subject_match) + \
                    (w.get("complexity", 0.3) * complexity_match) + \
                    (w.get("cost", 0.2) * -normalized_cost) + \
                    (w.get("latency", 0.2) * latency_penalty)

            # Tie-breaking: Higher priority wins if scores are mathematically equal
            if score > best_score or (abs(score - best_score) < 1e-5 and profile.get("priority", 0) > self.model_profiles.get(best_alias, {}).get("routing_profile", {}).get("priority", 0)):
                best_score, best_alias = score, alias

        # Graceful Degradation: If all models were filtered out (e.g. images present but no VLM), 
        # fall back to the highest priority text model.
        if not best_alias and self.child_bindings:
            best_alias = max(self.child_bindings.keys(), key=lambda a: self.model_profiles[a].get("routing_profile", {}).get("priority", 0))
            ASCIIColors.warning(f"[SmartRouter] No model matched hard filters. Falling back to highest priority: {best_alias}")

        return best_alias

    # ── Delegated LLM Methods ──
    def generate_text(self, prompt, *args, **kwargs):
        images = kwargs.get("images")
        chosen = self._select_model(prompt, images)
        if not chosen: raise RuntimeError("SmartRouter failed to select a suitable model and no fallback was available.")
        ASCIIColors.info(f"[SmartRouter] Routing text generation to: {chosen}")
        return self.child_bindings[chosen].generate_text(prompt, *args, **kwargs)

    def generate_from_messages(self, messages, *args, **kwargs):
        prompt_text = " ".join([m["content"] for m in messages if isinstance(m, dict) and m.get("role") in ["user", "system"]])
        images = kwargs.get("images")
        chosen = self._select_model(prompt_text, images)
        if not chosen: raise RuntimeError("SmartRouter failed to select a suitable model and no fallback was available.")
        ASCIIColors.info(f"[SmartRouter] Routing message generation to: {chosen}")
        return self.child_bindings[chosen].generate_from_messages(messages, *args, **kwargs)

    def tokenize(self, text): 
        if not self.child_bindings: return list(text)
        return list(self.child_bindings.values())[0].tokenize(text)

    def detokenize(self, tokens): 
        if not self.child_bindings: return "".join(tokens)
        return list(self.child_bindings.values())[0].detokenize(tokens)

    def count_tokens(self, text): 
        if not self.child_bindings: return len(text) // 4
        return list(self.child_bindings.values())[0].count_tokens(text)

    def get_ctx_size(self, model_name=None): 
        if not self.child_bindings: return 4096
        return max([b.get_ctx_size() for b in self.child_bindings.values()])

    def load_model(self, model_name): return True
    def list_models(self): return []

    def embed(self, text, **kwargs): 
        if not self.child_bindings: raise RuntimeError("No child bindings available for embedding.")
        return list(self.child_bindings.values())[0].embed(text, **kwargs)

    def get_model_info(self): return {"name": "SmartRouter", "type": "router"}