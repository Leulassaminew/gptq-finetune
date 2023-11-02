from datasets import load_dataset
from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training
from transformers import GPTQConfig
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

dataset = load_dataset("Leul78/final")
output_dir = "output_lora"

def format_instruction(sample):
	return f"""### Instruction:
"Categorize the input text based on the sales technique used in it from one of these categories only and offer no explanation:\n\nBUILDING RAPPORT\nNEEDS ASSESMENT\nCREATING URGENCY\nSOCIAL PROOF\nOVERCOMING OBJECTION\nCROSS SELLING OR UPSELLING\nVALUE BASED SELLING\nNONE\n\n"

### Input:
{sample['sentence']}

### Response:
{sample['label']}
"""

model_id = "TheBloke/Llama-2-13B-chat-GPTQ"

quantization_config_loading = GPTQConfig(bits=4, disable_exllama=True)
model = AutoModelForCausalLM.from_pretrained(
                              model_id,
                              device_map="auto"
                          )
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.config.use_cache = False
# https://github.com/huggingface/transformers/pull/24906
#disable tensor parallelism
model.config.pretraining_tp = 1

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
from peft import LoraConfig, get_peft_model
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["k_proj","o_proj","q_proj","v_proj"],
    lora_dropout=0.01,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=4 ,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="adamw_hf",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    disable_tqdm=True # disable tqdm since with packing values are in correct
)

max_seq_length = 512 # max sequence length for model and packing of the dataset

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    peft_config=config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=format_instruction,
    args=args,
)

trainer.train()
trainer.save_model()