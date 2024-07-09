from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
# from openai import OpenAI
import base64
import os
import time
from PIL import Image
import requests
import copy
import torch
# client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

pretrained = "lmms-lab/llama3-llava-next-8b"
model_name = "llava_llama3"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation=None) # Add any other thing you want to pass in llava_model_args

model.eval()
model.tie_weights()

def encode_image_from_url(file_path):
    image = Image.open(file_path)
    return image

def query_llava(image_urls, question, conv_template="llava_llama_3"):
    """
    Query the LLava model with the prompt and a list of image URLs.

    Parameters:
    - image_urls: List of Strings, the URLs to the images.
    - question: String, the question prompt.
    - conv_template: String, the conversation template to use.
    """
    images = [encode_image_from_url(image_url) for image_url in image_urls]
    image_tensor = process_images(images, image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
    image_sizes = [image.size for image in images]
    
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=256,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    return text_outputs



# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def query_gpt4v(image_paths, promt, retry=10):
   return "Not implemented yet"
# def query_gpt4v(image_paths, prompt, retry=10):
#     """
#     Query the GPT-4 Vision model with the prompt and a list of image paths. The temperature is set to 0.0 and retry is set to 10 if fails as default setting.

#     Parameters:
#     - image_paths: List of Strings, the path to the images.
#     - prompt: String, the prompt.
#     - retry: Integer, the number of retries.
#     """
#     base64_images = [encode_image(image_path) for image_path in image_paths]

#     for r in range(retry):
#         try:
#             input_dicts = [{"type": "text", "text": prompt}]
#             for i, image in enumerate(base64_images):
#                 input_dicts.append({"type": "image_url",
#                                     "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "low"}})
#             response = client.chat.completions.create(
#                 model="gpt-4-vision-preview",
#                 messages=[
#                     {
#                     "role": "user",
#                     "content": input_dicts,
#                     }
#                 ],
#                 max_tokens=1024,
#                 n=1,
#                 temperature=0.0,
#             )
#             print(response)
#             return response.choices[0].message.content
#         except Exception as e:
#             print(e)
#             time.sleep(1)
#     return 'Failed: Query GPT4V Error'
