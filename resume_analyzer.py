# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1QNqsNnm9loDBw_5G0RQ_hyMZWFgebGBh
"""

!pip install accelerate
!pip install langchain
!pip install langchain-community
!pip install transformers
!pip install sentence-transformers
!pip install faiss-cpu
!pip install unstructured
!pip install unstructured[pdf]

"""#Importing the Dependencies"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

"""#Loading the Model"""

torch.random.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

"""#Converting to Image"""

!pip install -U pypdfium2

import pypdfium2 as pdfium
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

def convert_pdf_to_images(file_path, scale=300/72):

    pdf_file = pdfium.PdfDocument(file_path)
    page_indices = [i for i in range(len(pdf_file))]

    renderer = pdf_file.render(
        pdfium.PdfBitmap.to_pil,
        page_indices = page_indices,
        scale = scale,
    )

    list_final_images = []

    for i, image in zip(page_indices, renderer):

        image_byte_array = BytesIO()
        image.save(image_byte_array, format='jpeg', optimize=True)
        image_byte_array = image_byte_array.getvalue()
        list_final_images.append(dict({i:image_byte_array}))

    return list_final_images

def display_images(list_dict_final_images):

    all_images = [list(data.values())[0] for data in list_dict_final_images]

    for index, image_bytes in enumerate(all_images):

        image = Image.open(BytesIO(image_bytes))
        figure = plt.figure(figsize = (image.width / 100, image.height / 100))

        plt.title(f"----- Page Number {index+1} -----")
        plt.imshow(image)
        plt.axis("off")
        plt.show()

convert_pdf_to_images = convert_pdf_to_images('/content/AryanShahResume.pdf')

display_images(convert_pdf_to_images)

"""#Converting IMG to text"""

!apt-get install tesseract-ocr
!pip install pytesseract

from pytesseract import image_to_string

def extract_text_with_pytesseract(list_dict_final_images):

    image_list = [list(data.values())[0] for data in list_dict_final_images]
    image_content = []

    for index, image_bytes in enumerate(image_list):

        image = Image.open(BytesIO(image_bytes))
        raw_text = str(image_to_string(image))
        image_content.append(raw_text)

    return "\n".join(image_content)

text_with_pytesseract = extract_text_with_pytesseract(convert_pdf_to_images)

"""#Defining a Prompt"""

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Go through the document and classify the sentences under headings given. Do not use any words out of English dictionary, if a word is unsure do not include it"},
    {"role": "user", "content": f"Here is the text {text_with_pytesseract}. Please ensure that no information from the document is missed during classification."},
    {"role": "user", "content": "Note: Do not miss contact details if in the text. Otherwise skip it, do not make up"},
    {"role": "user", "content": "Wherever it would be better make better groupings. Do not make any things up"},
    {"role": "assistant", "content": "Sure! Here is all the content in the text without missing anything."},
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 20000,
    "return_full_text": False,
    "temperature": 0,
}

output = pipe(messages, **generation_args)
response = output[0]['generated_text']
print(response)

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Can you extract entities out of the given text?."},
    {"role": "user", "content": f"Here is the text {text_with_pytesseract}. Do not miss any important things. Do not mess up important information."},
    {"role": "assistant", "content": "Sure! Here is all the content in the text without missing anything."},
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 20000,
    "return_full_text": False,
    "temperature": 0,
}

output = pipe(messages, **generation_args)
response = output[0]['generated_text']
print(response)

