"""LLM guard models implementation."""

from __future__ import annotations

import logging
from abc import abstractmethod
from textwrap import dedent
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TFPreTrainedModel,
)

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class LLMGuardModel(BaseModel):
    """LLM Guard models for hate speech detection."""

    def __init__(
        self,
        model_name: str,
        model_id: str | None,
        model: PreTrainedModel | TFPreTrainedModel | None,
        tokenizer: PreTrainedTokenizer | None,
        dtype: torch.dtype | None,
        device_map: str | None,
        device: str | None = None,
        random_seed: int | None = None,
        **kwargs,
    ):
        """Initialize LLM guard model.

        Args:
            model_name: Model name, case-insensitive.
            model_id: LLM model id or path
            device: Device for computation ('cpu', 'cuda', or None for auto)
            random_seed: Random seed for reproducibility
            **kwargs: Additional model arguments
        """
        # Initialize base class for device and seed management
        super().__init__(
            model_name=model_name,
            model_id=model_id,
            device=device,
            random_seed=random_seed,
            **kwargs,
        )

        self._model = model
        self._tokenizer = tokenizer
        self.dtype = dtype
        self.device_map = device_map
        self.kwargs = kwargs

    @property
    def model(self):
        """Underlying model instance."""
        return self._model

    @property
    def tokenizer(self):
        """Tokenizer instance (may be None until loaded)."""
        return self._tokenizer

    @model.setter
    def model(self, model: PreTrainedModel | TFPreTrainedModel):
        """Set the model instance."""
        self._model = model

    @tokenizer.setter
    def tokenizer(self, tokenizer: PreTrainedTokenizer):
        """Set the tokenizer instance."""
        self._tokenizer = tokenizer

    @abstractmethod
    def prepare_prompts(self, texts: list[str]) -> list[Any]:
        """Prepare prompts for LLM guard model.

        Args:
            texts: List of input texts

        Returns:
            Formatted prompts in model-specific format (e.g., list of strings,
            list of chat messages, etc.)
        """
        pass

    @abstractmethod
    def inference(self, texts: list[str], batch_size: int | None = 1) -> list[dict[str, Any]]:
        """Generate responses.

        Args:
            texts (list[str]): List of input texts
            batch_size (int, optional, defaults to 1): Batch size for processing.

        Returns:
            List of predictions with labels and probabilities
        """
        pass

    @abstractmethod
    def load(self) -> None:
        """set up model and tokenizer."""
        pass

    def free_memory(self) -> None:
        """Free model resources to release GPU memory.

        Clears model weights, tokenizer, and pipeline objects, then
        forces GPU cache cleanup and garbage collection.
        """
        try:
            # Clear model reference
            if hasattr(self, "_model") and self._model is not None:
                if hasattr(self._model, "cpu"):
                    self._model.cpu()
                self._model = None

            # Clear tokenizer reference
            if hasattr(self, "_tokenizer") and self._tokenizer is not None:
                self._tokenizer = None

            # Clear pipeline reference
            if hasattr(self, "pipeline") and self.pipeline is not None:
                del self.pipeline
                self.pipeline = None

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Force garbage collection
            import gc

            gc.collect()

        except Exception as e:
            logger.warning("Error during memory cleanup: %s", e)

    def train(self) -> None:
        pass

    def save(self) -> None:
        pass

    def __repr__(self):
        return f"""
        LLMGuardModel(model_id={self.model_id})
        """


class LlamaGuard(LLMGuardModel):
    def __init__(self, **kwargs):
        super().__init__(
            model_name="LlamaGuard",
            model_id="meta-llama/Llama-Guard-3-8B",
            model=kwargs.get("model"),
            tokenizer=kwargs.get("tokenizer"),
            dtype=kwargs.get("dtype"),
            device_map=kwargs.get("device_map", "auto"),
            **kwargs,
        )
        self.load()

    def load(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be provided")

        if not self._model:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=self.dtype,
                device_map=self.device_map,
            )

        if not self._tokenizer:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        if not (self.model and self.tokenizer):
            raise RuntimeError("model or tokenizer not loaded.")

        # override model configs
        self.model.config.pad_token_id = 128001
        self.tokenizer.pad_token_id = self.model.config.pad_token_id
        self.tokenizer.padding_side = "left"

    def prepare_prompts(self, texts: list[str]) -> list[list[dict[str, str]]]:
        """Prepare prompts for LLM guard model.

        Args:
            texts: List of input texts

        Returns:
            List of formatted prompts
        """
        return [[{"role": "user", "content": text}] for text in texts]

    def inference(self, texts: list[str], batch_size: int | None = 1) -> list[dict[str, Any]]:
        """
        Generate responses from the LLM guard model.
        Uses first token probability as "unsafe" class probability.
        Args:
            texts: List of input texts
            batch_size: Batch size for generation
        """
        # Ensure model and tokenizer are loaded
        assert self.model is not None, "Model not loaded. Call load() first."
        assert self.tokenizer is not None, "Tokenizer not loaded. Call load() first."

        inputs = self.prepare_prompts(texts)
        results = []

        # Handle None batch_size
        if batch_size is None:
            batch_size = 1

        # Process texts in batches
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i : i + batch_size]

            # Tokenize the batch
            encoded_inputs = self.tokenizer.apply_chat_template(
                batch_inputs, return_tensors="pt", padding=True
            )

            if hasattr(encoded_inputs, "to"):
                encoded_inputs = encoded_inputs.to(self.model.device)  # type: ignore[attr-defined]

            # Get logits from the model
            with torch.no_grad():
                outputs = self.model(encoded_inputs)
                logits = outputs.logits

                # Get the logits for the first token after the prompt
                first_token_logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]

                # Apply softmax to get probabilities
                first_token_probs = torch.softmax(first_token_logits, dim=-1)

                # Get token ID for "unsafe"
                unsafe_token_id = self.tokenizer.encode("unsafe", add_special_tokens=False)[0]

                # Extract probabilities for the unsafe token
                for j in range(first_token_logits.shape[0]):
                    # Use first token probability as "unsafe" class probability
                    unsafe_prob = first_token_probs[j, unsafe_token_id].item()
                    label = unsafe_prob > 0.5

                    results.append({"label": label, "prob": unsafe_prob})

        return results


class PromptGuard(LLMGuardModel):
    """Specific implementation for PromptGuard model."""

    def __init__(self, **kwargs):
        """Initialize PromptGuard model."""
        super().__init__(
            model_name="PromptGuard",
            model_id="meta-llama/Prompt-Guard-86M",
            model=kwargs.get("model"),
            tokenizer=kwargs.get("tokenizer"),
            dtype=kwargs.get("dtype"),
            device_map=kwargs.get("device_map", "auto"),
            **kwargs,
        )
        self.load()

    def load(self) -> None:
        """Load PromptGuard model and tokenizer."""
        if not self.model_id:
            raise ValueError("model_id must be provided")

        if not self._model:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                dtype=self.dtype,
                device_map=self.device_map,
            )

        if not self._tokenizer:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def prepare_prompts(self, texts: list[str]) -> list[str]:
        """Prepare prompts for PromptGuard - it works directly with text."""
        return texts

    def inference(self, texts: list[str], batch_size: int | None = 1) -> list[dict[str, Any]]:
        """
        Generate responses from the PromptGuard model for hate speech detection.
        Based on the original PromptGuard implementation.
        Args:
            texts: List of input texts
            batch_size: Batch size for generation
        """
        # Ensure model and tokenizer are loaded
        assert self.model is not None, "Model not loaded. Call load() first."
        assert self.tokenizer is not None, "Tokenizer not loaded. Call load() first."

        results = []

        # Handle None batch_size
        if batch_size is None:
            batch_size = 1

        # Get label mapping from model config for robustness
        label2id = getattr(self.model.config, "label2id", {})

        # Ensure label2id is a dictionary
        if not isinstance(label2id, dict):
            label2id = {}

        # Find JAILBREAK class ID from config, fallback to 2 if not found
        jailbreak_class_id = label2id.get("JAILBREAK")
        if jailbreak_class_id is None:
            # Try alternative names or fallback to default
            for label in ["jailbreak", "Jailbreak"]:
                if label in label2id:
                    jailbreak_class_id = label2id[label]
                    break
            else:
                # Fallback to 2 (original assumption) if not found in config
                jailbreak_class_id = 2

        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize the batch
            inputs = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
            )

            # Ensure all tensors in inputs are on the correct device
            if hasattr(inputs, "to"):
                inputs = inputs.to(self.model.device)
            elif isinstance(inputs, dict):
                inputs = {
                    k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()
                }

            # Get logits from the model
            with torch.no_grad():
                logits = self.model(**inputs).logits

                # Apply temperature scaling (using default temperature=1.0)
                temperature = 1.0
                scaled_logits = logits / temperature

                # Apply softmax to get probabilities
                probabilities = torch.softmax(scaled_logits, dim=-1)

                for j in range(logits.shape[0]):
                    # For hate speech detection, use JAILBREAK probability from config
                    jailbreak_prob = probabilities[j, jailbreak_class_id].item()
                    label = jailbreak_prob > 0.5

                    results.append({"label": label, "prob": jailbreak_prob})

        return results


class ShieldGemma(LLMGuardModel):
    """Specific implementation for ShieldGemma model."""

    def __init__(self, **kwargs):
        """Initialize ShieldGemma model."""
        super().__init__(
            model_name="ShieldGemma",
            model_id="google/shieldgemma-2b",
            model=kwargs.get("model"),
            tokenizer=kwargs.get("tokenizer"),
            dtype=kwargs.get("dtype"),
            device_map=kwargs.get("device_map", "auto"),
            **kwargs,
        )
        self.load()

    def load(self) -> None:
        """Load ShieldGemma model and tokenizer."""
        if not self.model_id:
            raise ValueError("model_id must be provided")

        # if model not provided, load from huggingface
        if not self._model:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=self.dtype,
                device_map=self.device_map,
            )
        # if tokenizer not provided, load from huggingface
        if not self._tokenizer:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def prepare_prompts(self, texts: list[str]) -> list[str]:
        """
        Prepare prompts for ShieldGemma using hate speech detection template.
        Based on ShieldGemma documentation for hate speech detection.
        """

        def chat_template(user_prompt: str) -> str:
            prompt = f"""
                You are a policy expert trying to help determine whether a user prompt is in
                violation of the defined safety policies.

                <start_of_turn>
                Human Question: {user_prompt}
                <end_of_turn>

                Our safety principle is defined in the below:

                * "No Hate Speech": The prompt shall not contain or seek generation of content
                that expresses, incites, or promotes hate based on race, gender, ethnicity,
                religion, nationality, sexual orientation, disability status, or caste.

                ===

                Does the human question violate the above principle? Your answer must start with
                'Yes' or 'No'. And then walk through step by step to be sure we answer
                correctly.
            """
            return prompt

        prompts = [dedent(chat_template(text)).strip() for text in texts]
        return prompts

    def inference(self, texts: list[str], batch_size: int | None = 1) -> list[dict[str, Any]]:
        """
        Generate responses from the ShieldGemma model for hate speech detection.
        Based on the HuggingFace implementation pattern.
        Args:
            texts: List of input texts
            batch_size: Batch size for generation
        """
        # Ensure model and tokenizer are loaded
        assert self.model is not None, "Model not loaded. Call load() first."
        assert self.tokenizer is not None, "Tokenizer not loaded. Call load() first."

        prompts = self.prepare_prompts(texts)
        results = []

        # Handle None batch_size
        if batch_size is None:
            batch_size = 1

        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]

            # Tokenize the batch
            inputs = self.tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True
            )

            if hasattr(inputs, "to"):
                inputs = inputs.to(self.model.device)
            elif isinstance(inputs, dict):
                inputs = {
                    k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()
                }
            # Get logits from the model
            with torch.no_grad():
                logits = self.model(**inputs).logits

                # Extract the logits for the Yes and No tokens (following official guide)
                vocab = self.tokenizer.get_vocab()

                # Get Yes and No token IDs
                yes_token_id = vocab.get("Yes")
                no_token_id = vocab.get("No")

                if yes_token_id is not None and no_token_id is not None:
                    # Extract logits for Yes and No tokens from last position
                    selected_logits = logits[:, -1, [yes_token_id, no_token_id]]

                    # Convert to probabilities with softmax
                    probabilities = torch.softmax(selected_logits, dim=-1)

                    for j in range(probabilities.shape[0]):
                        # Probability of 'Yes' (violates policy = hate speech)
                        yes_prob = probabilities[j, 0].item()
                        label = yes_prob > 0.5

                        results.append({"label": label, "prob": yes_prob})
                else:
                    # Simple fallback: if Yes/No tokens not found, assume "No" (safe default)
                    logger.warning(
                        "Yes/No tokens not found in vocabulary. "
                        "Defaulting to 'No' (non-hate) for safety."
                    )
                    for j in range(logits.shape[0]):
                        # Default to "No" violation (prob=0.0, label=False)
                        results.append({"label": False, "prob": 0.0})

        return results
