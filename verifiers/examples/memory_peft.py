import dotenv
dotenv.load_dotenv()

from trl import GRPOTrainer
import verifiers.verifiers as vf
from peft import LoraConfig, get_peft_model

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

# Configure PEFT (LoRA)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# Apply PEFT to the model
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # Log the trainable parameters

vf_env = vf.MemoryToolEnv(dataset="memory")
dataset = vf_env.get_dataset()
rubric = vf_env.get_rubric()
training_args = vf.get_default_grpo_config(run_name="memory_peft_dria_agent", num_gpus=1)
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric, 
    env=vf_env,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()