import verifiers as vf
from verifiers.tools import search_web
from verifiers.prompts import SEARCH_FEW_SHOT
from peft import LoraConfig, get_peft_model

model_name = "Qwen/Qwen2.5-7B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

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

vf_env = vf.ToolEnv(
    dataset="search",
    #few_shot=SEARCH_FEW_SHOT[0],
    tools=[search_web],
    max_steps=3
)
train_dataset = vf_env.get_dataset()
rubric = vf_env.get_rubric()

# notable defaults: lr = 1e-6, max_grad_norm = 0.01, constant lr 10 warmup steps, 1024 tokens in+out
run_name = "search_peft" + model_name.split("/")[-1].lower()
training_args = vf.get_default_grpo_config(
    run_name=run_name,
    num_gpus=8 # 7 train + 1 inference
)
# rollouts per prompt
training_args.num_generations = 6  # Changed from 7 to 6 to make it evenly divide the batch size
# minibatch size per GPU ( bs 6 * 7 gpus / 6 rollouts -> 7 prompts per batch)
training_args.per_device_train_batch_size = 6
# batches to accumulate (6 prompts * 4 -> 24 prompts per global batch)
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

