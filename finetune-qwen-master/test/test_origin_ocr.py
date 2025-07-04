# import some libraries
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from functools import partial

from swift.llm import (
    EncodePreprocessor,
    LazyLLMDataset,
    get_model_arch,
    get_model_tokenizer,
    get_multimodal_target_regex,
    get_template,
    load_dataset,
)
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from swift.tuners import LoraConfig, Swift
from swift.utils import (
    get_logger,
    get_model_parameter_info,
    plot_images,
    seed_everything,
)

logger = get_logger()
seed_everything(42)

# Hyperparameters for training
# model
model_id_or_path = "Qwen/Qwen2.5-VL-3B-Instruct"
system = None  # Using the default system defined in the template.
output_dir = "output"

# dataset
dataset = ["AI-ModelScope/LaTeX_OCR#20000"]  # dataset_id or dataset_path. Sampling 20000 data points
data_seed = 42
max_length = 2048
split_dataset_ratio = 0.01  # Split validation set
num_proc = 1  # The number of processes for data loading.

# lora
lora_rank = 8
lora_alpha = 32
freeze_llm = False
freeze_vit = True
freeze_aligner = True

# training_args
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    report_to=["tensorboard"],
    logging_first_step=True,
    save_strategy="steps",
    save_steps=50,
    eval_strategy="steps",
    eval_steps=50,
    # To observe the training results more quickly, this is set to 1 here.
    # Under normal circumstances, a larger number should be used.
    num_train_epochs=1,
    metric_for_best_model="loss",
    save_total_limit=5,
    logging_steps=5,
    dataloader_num_workers=1,
    data_seed=data_seed,
    remove_unused_columns=False,
)

output_dir = os.path.abspath(os.path.expanduser(output_dir))
logger.info(f"output_dir: {output_dir}")

# Obtain the model and template
model, processor = get_model_tokenizer(model_id_or_path)
logger.info(f"model_info: {model.model_info}")
template = get_template(model.model_meta.template, processor, default_system=system, max_length=max_length)
template.set_mode("train")

# Get target_modules and add trainable LoRA modules to the model.
model_arch = get_model_arch(model.model_meta.model_arch)
target_modules = get_multimodal_target_regex(
    model_arch, freeze_llm=freeze_llm, freeze_vit=freeze_vit, freeze_aligner=freeze_aligner
)
lora_config = LoraConfig(task_type="CAUSAL_LM", r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
model = Swift.prepare_model(model, lora_config)
logger.info(f"lora_config: {lora_config}")

# Print model structure and trainable parameters.
logger.info(f"model: {model}")
model_parameter_info = get_model_parameter_info(model)
logger.info(f"model_parameter_info: {model_parameter_info}")

# Download and load the dataset, split it into a training set and a validation set,
# and encode the text data into tokens.
train_dataset, val_dataset = load_dataset(
    dataset, split_dataset_ratio=split_dataset_ratio, num_proc=num_proc, seed=data_seed
)

logger.info(f"train_dataset: {train_dataset}")
logger.info(f"val_dataset: {val_dataset}")
logger.info(f"train_dataset[0]: {train_dataset[0]}")

train_dataset = LazyLLMDataset(train_dataset, template.encode, random_state=data_seed)
val_dataset = LazyLLMDataset(val_dataset, template.encode, random_state=data_seed)
data = train_dataset[0]
logger.info(f"encoded_train_dataset[0]: {data}")

template.print_inputs(data)

# Get the trainer and start the training.
model.enable_input_require_grads()  # Compatible with gradient checkpointing
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=template.data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    template=template,
)
trainer.train()
