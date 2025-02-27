import dotenv
dotenv.load_dotenv()

from trl import GRPOTrainer
import verifiers as vf

model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

# Use the new tool-based memory environment
vf_env = vf.MemoryToolEnvNew(dataset="memory")
dataset = vf_env.get_dataset()
rubric = vf_env.get_rubric()
training_args = vf.get_default_grpo_config(run_name="memory_tool_env", num_gpus=1)
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric, 
    env=vf_env,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()