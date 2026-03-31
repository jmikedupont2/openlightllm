"""Petals — decentralized LLM inference provider for litellm."""
import litellm

def completion(model, messages, **kwargs):
    """Route to petals distributed inference."""
    try:
        from petals import AutoDistributedModelForCausalLM
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model)
        model_obj = AutoDistributedModelForCausalLM.from_pretrained(model)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model_obj.generate(**inputs, max_new_tokens=kwargs.get("max_tokens", 256))
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return litellm.ModelResponse(choices=[{"message": {"content": text}}])
    except ImportError:
        raise litellm.exceptions.BadRequestError("petals not installed: pip install petals")
