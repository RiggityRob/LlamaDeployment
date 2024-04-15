---------------------------------------------------------------------------------------------------

https://github.com/ggerganov/llama.cpp/releases

so deploying a local LLM is kind of easy now lol

not just with Chat with RTX

but theres tons of ports of Llama for different hardware (AVX512, Vulkan, Cuda,  Intel GPU/SYCL, ARM64 support?)

and it supports a lot of different models

Im having to mess around with AMD friendly stuff on this laptop (deliberate choice).
but definitely best/easiest will be CUDA on Nvidia hardware

Gonna try and do a write up if you want to try it later

all the different extensions means its using different math to run the same LLM (edited) 

the big magic here is the quantization, or how the LLM is compressed and converted to diff maths. And this also frees us from the GPU shackles (it will perform worse tho)

still incredibly impressive that its been "easy" to port it into so many instruction sets.

lol okay idk if I even need to make like a formal document

with git, python3, and pip you can do

CMAKE_ARGS="-DLLAMA_VULKAN=on" pip install llama-cpp-python

download a model from https://huggingface.co/models?other=LLM&sort=downloads or use the link below

then in a .py file, call the model

from llama_cpp import Llama


# GLOBAL VARIABLES
my_model_path = "./zephyr-7b-beta.Q4_0.gguf"
CONTEXT_SIZE = 512


# LOAD THE MODEL
zephyr_model = Llama(model_path=my_model_path,
                    n_ctx=CONTEXT_SIZE)

and make a wrapper to give it input

def generate_text_from_prompt(user_prompt,
                             max_tokens = 100,
                             temperature = 0.3,
                             top_p = 0.1,
                             echo = True,
                             stop = ["Q", "\n"]):




   # Define the parameters
   model_output = zephyr_model(
       user_prompt,
       max_tokens=max_tokens,
       temperature=temperature,
       top_p=top_p,
       echo=echo,
       stop=stop,
   )


   return model_output

and call that wrapper to get responses like

if __name__ == "__main__":


   prompt = input("Uhhhhh:")


   model_response = generate_text_from_prompt(prompt)


   print(model_response)

(edited)

(switched this for a direct link, its a 4GB download)
https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_0.gguf?ref=localhost
heres the model used in the example, I did a Vulkan install to  be hardware agnostic but theres other options

heres the files
2 files

and use the gguf/localhost link above to download the exact model in the config.

This should be a Vulkan deployment so it will work on any GPU

when "Uhhhh:" is printed it is ready to accept a prompt

---------------------------------------------------------------------------------------------------
