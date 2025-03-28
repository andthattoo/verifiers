import verifiers as vf
from verifiers.tools import search, scrape
from verifiers.prompts import SEARCH_FEW_SHOT
from peft import LoraConfig, get_peft_model

model_name = "Qwen/Qwen2.5-7B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

peft_config = LoraConfig(
    r=128,
    lora_alpha=256,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# Apply PEFT to the model
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # Log the trainable parameters

vf_env = vf.ToolEnv(
    dataset="search",
    #few_shot=SEARCH_FEW_SHOT[1],
    tools=[search],
    max_steps=10
)
train_dataset = vf_env.get_dataset()
rubric = vf_env.get_rubric()

# notable defaults: lr = 1e-6, max_grad_norm = 0.01, constant lr 10 warmup steps, 1024 tokens in+out
run_name = "search_peft_" + model_name.split("/")[-1].lower()
training_args = vf.get_default_grpo_config(
    run_name=run_name,
    num_gpus=1 # Single GPU setup
)
# rollouts per prompt
training_args.num_generations = 6  # Changed from 7 to 6 to make it evenly divide the batch size
# minibatch size per GPU (bs 6 on single GPU / 6 rollouts -> 1 prompt per batch)
training_args.per_device_train_batch_size = 6
# batches to accumulate (1 prompt * 4 -> 4 prompts per global batch)
training_args.gradient_accumulation_steps = 4
# steps per global batch (1 on-policy, 2 off-policy)
training_args.num_iterations = 2
# no ref model
training_args.beta = 0.0
trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric,
    env=vf_env,
    args=training_args,
    train_dataset=train_dataset
)
trainer.train()

