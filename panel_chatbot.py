# # Dependency
# pip install huggingface-hub
# pip install panel
# #If you have no graphic card use this:
# pip install ctransformers
# #If you have a graphic card use this:
# pip install ctransformers[cuda]

import panel as pn
from ctransformers import AutoConfig, AutoModelForCausalLM, Config

pn.extension()
llms = {}

#This is Instruction how your bot shoud behave
SYSTEM_INSTRUCTIONS = "You are a friendly chat bot willing to help out the user."

#This is the memory/context function
def apply_template(history):
    history = [message for message in history if message.user != "System"]
    prompt = ""
    for i, message in enumerate(history):
        if i == 0:
            prompt += f"<s>[INST]{SYSTEM_INSTRUCTIONS} {message.object}[/INST]"
        else:
            if message.user == "Mistral":
                prompt += f"{message.object}</s>"
            else:
                prompt += f"""[INST]{message.object}[/INST]"""
    return prompt

# This calls the local LLM
async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    if "mistral" not in llms:
        # The first time the model is loaded 
        # this can take some time because it is 4.37G
        instance.placeholder_text = "Downloading model; please wait..."
        config = AutoConfig(
            config=Config(
                temperature=0.5, max_new_tokens=2048, context_length=2048, gpu_layers=40
            ),
        )
        llms["mistral"] = AutoModelForCausalLM.from_pretrained(
            "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
            model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            config=config,
        )

    llm = llms["mistral"]
    history = [message for message in instance.objects]
    prompt = apply_template(history)
    response = llm(prompt, stream=True)
    message = ""
    for token in response:
        message += token
        yield message

#This creates the complete ChatInteface
chat_interface = pn.chat.ChatInterface(
    callback=callback,
    callback_user="Mistral",
)

chat_interface.send(
    "Send a message to get a reply from Mistral!", user="System", respond=False
)
chat_interface.servable()