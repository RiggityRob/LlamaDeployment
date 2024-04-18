from llama_cpp import Llama


# GLOBAL VARIABLES
my_model_path = "./deepseek-coder-6.7b-instruct.Q6_K.gguf"
CONTEXT_SIZE = 2048


# LOAD THE MODEL
init_model = Llama(model_path=my_model_path, n_ctx=CONTEXT_SIZE, n_gpu_layers=2, chat_format='llama-2')


def generate_text_from_prompt(
    user_prompt, max_tokens=1000, temperature=0.3, top_p=0.1, echo=True, stop=["Q"]
):

    # Define the parameters
    model_output = init_model(
        user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        echo=echo,
        stop=stop,
    )

    return model_output


if __name__ == "__main__":

    f = open('responses.txt', 'a')

    prompt = input("Uhhhhh:")
    
    while prompt != "0":

        model_response = generate_text_from_prompt(prompt) 

        print(model_response["choices"][0]["text"].replace('\n','\n \r'))

        print(model_response["choices"][0]["text"].replace('\n','\n \r'), file=f)

        prompt = input("Uhhhhh:")

    f.close
