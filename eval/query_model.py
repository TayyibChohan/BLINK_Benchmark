# for lava:
# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
# from llava.conversation import conv_templates, SeparatorStyle
# from openai import OpenAI
# for qwenvl2:
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import base64
import os
import time
from PIL import Image
import requests
import copy
import torch
# client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# pretrained = "lmms-lab/llama3-llava-next-8b"
# model_name = "llava_llama3"
# device = "cuda"
# device_map = "auto"
# tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation=None) # Add any other thing you want to pass in llava_model_args

# model.eval()
# model.tie_weights()


def encode_image_from_url(file_path):
    image = Image.open(file_path)
    return image


def query_llava(image_urls, question, conv_template="llava_llama_3"):
    return "Not implemented yet"
# def query_llava(image_urls, question, conv_template="llava_llama_3"):
#     """
#     Query the LLava model with the prompt and a list of image URLs.

#     Parameters:
#     - image_urls: List of Strings, the URLs to the images.
#     - question: String, the question prompt.
#     - conv_template: String, the conversation template to use.
#     """
#     images = [encode_image_from_url(image_url) for image_url in image_urls]
#     image_tensor = process_images(images, image_processor, model.config)
#     image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
#     image_sizes = [image.size for image in images]

#     conv = copy.deepcopy(conv_templates[conv_template])
#     conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
#     conv.append_message(conv.roles[1], None)
#     prompt_question = conv.get_prompt()

#     input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

#     cont = model.generate(
#         input_ids,
#         images=image_tensor,
#         image_sizes=image_sizes,
#         do_sample=False,
#         temperature=0,
#         max_new_tokens=256,
#     )
#     text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
#     print(text_outputs)

#     return "".join(text_outputs)


model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype="auto",
    # torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    device_map="auto",
)

min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

def generate_qwen_vl2_message(image_paths, prompt, format_as_json=True):
        """
        Generate the message for the QwenVL2 model. in the following format:
         messages = [
                 {
                     "role": "user",
                     "content": [
                         {"type": "image", "image": "data:image;base64,/9j/..."},
                         {"type": "text", "text": "Describe this image."},
                     ],
                 }
             ]

         Parameters:
         - image_paths: Base64 encoded image paths
         - prompt: String, the prompt
        """
        messages = []
        content = []
        for image_path in image_paths:
            content.append({"type": "image", "image": f"data:image;base64,{image_path}"})
        if not format_as_json:
            content.append({"type": "text", "text": prompt})
        else:
            text = prompt + "\n" + "Format your answer as a JSON object with the following keys: 'answer', 'explanation'."
            content.append({"type": "text", "text": text})
        messages.append({"role": "user", "content": content})
        return messages

def query_qwenvl2(image_paths, prompt, retry=10):
    """
    Query the QwenVL2 model with the prompt and a list of image paths.

    Parameters:
    - image_paths: List of Strings, the path to the images.
    - prompt: String, the prompt.
    - retry: Integer, the number of retries.
    """
    # print(prompt)
    for r in range(retry):
        try:
            base64_images = [encode_image(image_path) for image_path in image_paths]
            messages = generate_qwen_vl2_message(base64_images, prompt)
            text = processor.apply_chat_template(messages)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            print(output_text)
            return output_text
        except Exception as e:
            print(e)
            time.sleep(1)
    return 'Failed: Query QwenVL2 Error'


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
