from ctransformers import AutoModelForCausalLM, AutoConfig, Config
# Set gpu_layers to the number of layers to offload to GPU. 
# Set to 0 if no GPU acceleration is available on your system. Default = 0

config = AutoConfig(
            config=Config(
                temperature=0.5, max_new_tokens=2048, context_length=2048, gpu_layers=20
            ),
        )

# Mistral
#llm = AutoModelForCausalLM.from_pretrained(".\models\Mistral-7B-Instruct-v0.1-GGUF", model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf", model_type="mistral", gpu_layers=0)
llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-GGUF", model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf", config=config)
# LLama
#llm = AutoModelForCausalLM.from_pretrained(".\models\Llama-2-7B-Chat-GGUF", model_file="llama-2-7b-chat.Q4_K_M.gguf", model_type="llama", gpu_layers=20)
# Zephyr
#llm = AutoModelForCausalLM.from_pretrained(".\models\zephyr-7B-beta-GGUF", model_file="zephyr-7b-beta.Q4_K_M.gguf", model_type="zephyr", gpu_layers=20)


prompt = "What means singularity for humans?"
for text in llm(prompt, stream=True):
    print(text, end="", flush=True)