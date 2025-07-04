from swift.llm.argument import InferArguments
from swift.llm.infer.utils import prepare_adapter
from swift.llm.model.register import get_model_tokenizer

from .qwen_forensics_model import QwenForensicModel


def prepare_model_template(args: InferArguments, **kwargs):
    kwargs.update(args.get_model_kwargs())
    model, processor = get_model_tokenizer(**kwargs)

    model: QwenForensicModel = model

    model.setup_tokens_for_conversation(tokenizer=processor.tokenizer)

    model = prepare_adapter(args, model)
    template = args.get_template(processor)
    return model, template
