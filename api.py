from flask import Flask, request, jsonify, Response
from marshmallow import Schema, fields, ValidationError, post_load, validate
import modules.async_worker as worker
import modules.advanced_parameters as advanced_parameters
import modules.style_sorter as style_sorter
import shared
from modules.sdxl_styles import legal_style_names

import time
import json
import sys
import numpy as np

from flask_cors import CORS

from PIL import Image
import io
import base64

#gross
print('[System ARGV] ' + str(sys.argv))
temp = sys.argv
sys.argv = ['launch_from_api.py', '--preset', 'anime']
from launch_from_api import *
import modules.config as config
sys.argv = temp
#

app = Flask(__name__)
CORS(app)


@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route('/json', methods=['POST'])
def json_post():
    # json keys must be strings
    return {
        "this": "is",
        "json": "kindof"
    }

#TODO: getStyles, getAspectRatios, getModels
@app.route('/styles')
def get_styles():
    if len(style_sorter.all_styles ) == 0:
        style_sorter.try_load_sorted_styles(
            style_names=legal_style_names,
            default_selected=config.default_styles)
    return style_sorter.all_styles

@app.route('/styles/default')
def get_default_styles():
    return config.default_styles

@app.route('/aspect-ratios')
def get_aspect_ratios():
    available = config.available_aspect_ratios
    aspect_ratios = []
    for ratio in available:
        split = ratio.split("<")
        dimensions = split[0].strip()
        aspect_ratio = split[1].split(">")[1].strip()[1:].strip()
        aspect_ratios.append({ "dimensions": dimensions, "ratio": aspect_ratio })

    return jsonify(aspect_ratios)

@app.route('/models')
def get_models():
    return config.get_model_filenames(config.path_checkpoints)

#TODO: make this update 
@app.route('/loras')
def get_loras():
    config.update_all_model_names()
    return config.lora_filenames

#TODO: if this works there needs to be some sort of ID associated with each 
@app.route('/image/gen/cancel')
def cancel_image_gen():
    import ldm_patched.modules.model_management as model_management
    shared.last_stop = 'stop'
    model_management.interrupt_current_processing()
    return "success"

@app.route('/image/gen/skip')
def skip_image_gen():
    import ldm_patched.modules.model_management as model_management
    shared.last_stop = 'skip'
    model_management.interrupt_current_processing()
    return "success"

class LoraSchema(Schema):
    model = fields.Str(required=True)
    weight = fields.Float(required=True) 

class ImageGenSchema(Schema):
    # name = fields.Str(required=True, validate=lambda s: 4 <= len(s) <= 25)
    # age = fields.Int(required=True, validate=lambda n: 0 <= n <= 100)

    # loras_parameters = fields.List(fields.List(fields.Raw()), required=False, data_key="lorasParameters")
    lora_parameters = fields.List(fields.Nested(LoraSchema), required=False, data_key="loraParameters" , validate=validate.Length(max=5))
    refiner_switch = fields.Float(required=False, data_key="refinerSwitch")
    refiner_model_name = fields.Str(required=False, data_key="refinerModelName")
    base_model_name = fields.Str(required=True, data_key="baseModelName")
    guidance_scale = fields.Int(required=False, data_key="guidanceScale")
    sharpness = fields.Int(required=False, data_key="sharpness")
    image_seed = fields.Str(required=True, data_key="imageSeed")
    image_number = fields.Int(required=False, data_key="imageNumber", validate=lambda n: 0 < n <= 24)
    aspect_ratios_selection = fields.Str(required=True, data_key="aspectRatiosSelection")
    performance_selection = fields.Str(
        required=True, 
        data_key="performanceSelection", 
        validate=validate.OneOf(["Speed", "Quality", "Extreme Speed"])
    )
    style_selections = fields.List(fields.Str(), required=False, data_key="styleSelections")
    negative_prompt = fields.Str(required=False, data_key="negativePrompt")
    prompt = fields.Str(required=True, data_key="prompt")
    enable_preview_images = fields.Bool(required=False, data_key="enablePreviewImages")

    @post_load
    def make_snake_case(self, data, **kwargs):
        # Convert camelCase keys to snake_case
        return {self.camel_to_snake(key): value for key, value in data.items()}

    @staticmethod
    def camel_to_snake(name):
        import re
        str1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', str1).lower()

#TODO: stop and skip
#TODO: return expanded prompt
@app.route('/image/gen', methods=['POST'])
def image_gen():
    print("Image gen endpoint")

    schema = ImageGenSchema()
    try:
        data = schema.load(request.json)

        number_of_images = data.get('image_number') if data.get('image_number') else 1
        enable_preview_images = data.get('enable_preview_images') if data.get('enable_preview_images') else False
        
        config = [
            data.get('prompt'),
            data.get('negative_prompt') if data.get('negative_prompt') else '',
            data.get('style_selections') if data.get('style_selections') else [], 
            data.get('performance_selection'),
            data.get('aspect_ratios_selection'), #'1152×896', #TODO: aspect_ratio '1152×896 <span style="color: grey;"> ∣ 9:7</span>',
            number_of_images,
            data.get('image_seed'),
            data.get('sharpness') if data.get('sharpness') else 2,
            data.get('guidance_scale') if data.get('guidance_scale') else 7,
            data.get('base_model_name'),
            data.get('refiner_model_name') if data.get('refiner_model_name') else "None", #note this is the string "None" not the type None
            data.get('refiner_switch') if data.get('refiner_switch') else 0.667,
            False, # data.get('input_image_checkbox')
            "uov", # data.get('current_tab')
            "Disabled", # data.get('uov_method')
            None, # data.get('uov_input_image')
            "[]", # data.get('outpaint_selections')
            None, # data.get('inpaint_input_image')
            "", # data.get('inpaint_additional_prompt')
            #TODO: learn what these are
            None, 0.5, 0.6, 'ImagePrompt', None, 0.5, 0.6, 'ImagePrompt', None, 0.5, 0.6, 'ImagePrompt', None, 0.5, 0.6, 'ImagePrompt'
        ]

        lora_params = data.get('lora_parameters') if data.get('lora_parameters') else []
        while (len(lora_params) < 5):
            lora_params.append({ "model":"None", "weight": 0 })
        # lora_config = [element for item in lora_params for element in (item['model'], item['weight'])]
        lora_config = []
        for item in lora_params:
            lora_config.extend([item['model'], item['weight']])
        
        print("----------------- LORA CONFIG -----------------------------")
        print(lora_config)

        # lora_config = ['sd_xl_offset_example-lora_1.0.safetensors', 0.5, 'None', 1, 'None', 1, 'None', 1, 'None', 1]
        lora_config_index = 12
        values = config[:lora_config_index] + lora_config + config[lora_config_index:]


        print("Attempting to gen image with values: ", values)

        # need to call set_all_advanced_parameters before running a task
        advanced_parameters.set_all_advanced_parameters(False, 1.5, 0.8, 0.3, 7, 'dpmpp_2m_sde_gpu', 'karras', False, -1, -1, -1, -1, -1, -1, False, False, False, False, 0.25, 64, 128, 'joint', False, 1.01, 1.02, 0.99, 0.95, False, False, 'v2.6', 1, 0.618)
        # task = worker.AsyncTask(args=params)
        task = worker.AsyncTask(args=values)

        worker.async_tasks.append(task)

        print('taks queued')

        


        # print("---image/gen endpoint results: ", task.results)

        return Response(response_stream(task, number_of_images, enable_preview_images), mimetype='text/event-stream')

        return jsonify("Horrya")
    except ValidationError as err:
        return jsonify(err.messages), 400
    
    return jsonify({"message": "Something unexpected happened"}), 500

def response_stream(task, number_of_images, enable_preview_images):
    data = { "updateType":"init", "percentage": 1, "title": 'Waiting for task to start ...', }
    json_data = json.dumps(data)
    yield f"{json_data}\n\n"

    image_results_sent = 0

    finished = False
    while not finished:
        time.sleep(0.01)
        if len(task.yields) > 0:
            flag, product = task.yields.pop(0)
            if flag == 'preview':

                # help bad internet connection by skipping duplicated preview
                if len(task.yields) > 0:  # if we have the next item
                    if task.yields[0][0] == 'preview':   # if the next item is also a preview
                        # print('Skipped one preview for better internet connection.')
                        continue

                percentage, title, image = product
                print ('NOT FINISHED, preview', percentage, title)
                # yield (percentage, title)
                if (enable_preview_images):

                    image_url = ""
                    height = 0
                    width = 0

                    if isinstance(image, np.ndarray):
                        print("Image shape: ", image.shape)
                        image_url = encode_image(image)
                        height = image.shape[0]
                        width = image.shape[1]
                        # image = image.tolist()
                    data = { 
                        "updateType": "preview", 
                        "percentage": percentage, 
                        "title": title, 
                        # "image": image,
                        "imageData": { 
                            "imageUrl": image_url,
                            "height": height,
                            "width": width
                        }
                    }
                    json_data = json.dumps(data)
                    yield f"{json_data}\n\n"
                else:
                    data = { 
                        "updateType": "preview", 
                        "percentage": percentage, 
                        "title": title, 
                        }
                    json_data = json.dumps(data)
                    yield f"{json_data}\n\n"

            if flag == 'results':
                print ('RESULTS, results')
                print(f"The type of 'product' is: {type(product)}")

                image_url = ""
                height = 0
                width = 0

                if isinstance(product, np.ndarray):
                    image_url = encode_image(product)

                    print("Product shape: ", product.shape)
                    # product = product.tolist()
                    
                elif isinstance(product, list):
                    image_url = encode_image(product[image_results_sent])
                    height = product[image_results_sent].shape[0]
                    width = product[image_results_sent].shape[1]
                    # Convert each ndarray element in the list to a list
                    # product = [item.tolist() if isinstance(item, np.ndarray) else item for item in product]
                    # product = product[image_results_sent].tolist()
                # else:
                #     product = product
                data = {
                    "updateType": "results", 
                    # "product": product,
                    "imageData": {
                        "imageUrl": image_url,
                        "height": height,
                        "width": width
                    }
                }
                json_data = json.dumps(data)
                image_results_sent = image_results_sent + 1
                yield f"{json_data}\n\n"

            if flag == 'finish':
                print ('FINISHED')
                print(f"The type of 'product' is: {type(product)}")

                image_url = ""
                height = 0
                width = 0
                image_urls = []

                if isinstance(product, np.ndarray):
                    image_url = encode_image(product)
                    height = product.shape[0]
                    width = product.shape[1]
                    print("Product shape: ", product.shape)
                    # product = product.tolist()
                elif isinstance(product, list):
                    print("PRODUCT LENGTH: ", )
                    image_urls = [{ "imageUrl": encode_image(item), "height": item.shape[0], "width": item.shape[1] } if isinstance(item, np.ndarray) else "" for item in product]
                    # Convert each ndarray element in the list to a list
                    # product = [item.tolist() if isinstance(item, np.ndarray) else item for item in product]
                # else:
                #     image_url = encode_image(product)
                    # product = product
                
                data
                if len(product) == 1:
                    data = {
                        "updateType": "finished", 
                        # "products": product,
                        "imagesData": image_urls   
                    }
                else:
                    data = {
                        "updateType": "finished", 
                        # "products": product
                    }
                json_data = json.dumps(data)
                yield f"{json_data}\n\n"
                finished = True

def encode_image(image):
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)

        # Save the image to a buffer
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")  # You can change the format to JPEG or other types
        buffer.seek(0)

        # Encode the image in base64
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        # Create the Data URL
        return f"data:image/png;base64,{img_base64}"
    else:
        raise Exception("image must be num py array")


# @app.route('/image/gen/2', methods=['POST'])
# def process_params_2():
#     print("Image gen endpoint")

#     schema = ImageGenSchema()
#     try:
#         data = schema.load(request.json)
        
#         config = [
#             data.get('prompt'),
#             data.get('negative_prompt') if data.get('negative_prompt') else '',
#             data.get('style_selections') if data.get('style_selections') else [], 
#             data.get('performance_selection'),
#             data.get('aspect_ratios_selection'), #'1152×896', #TODO: aspect_ratio '1152×896 <span style="color: grey;"> ∣ 9:7</span>',
#             data.get('image_number') if data.get('image_number') else 1,
#             data.get('image_seed'),
#             data.get('sharpness') if data.get('sharpness') else 2,
#             data.get('guidance_scale') if data.get('guidance_scale') else 7,
#             data.get('base_model_name'),
#             data.get('refiner_model_name') if data.get('refiner_model_name') else "None", #note this is the string "None" not the type None
#             data.get('refiner_switch') if data.get('refiner_switch') else 0.667,
#             False, # data.get('input_image_checkbox')
#             "uov", # data.get('current_tab')
#             "Disabled", # data.get('uov_method')
#             None, # data.get('uov_input_image')
#             "[]", # data.get('outpaint_selections')
#             None, # data.get('inpaint_input_image')
#             "", # data.get('inpaint_additional_prompt')
#             #TODO: learn what these are
#             None, 0.5, 0.6, 'ImagePrompt', None, 0.5, 0.6, 'ImagePrompt', None, 0.5, 0.6, 'ImagePrompt', None, 0.5, 0.6, 'ImagePrompt'
#         ]

#         lora_config = ['sd_xl_offset_example-lora_1.0.safetensors', 0.5, 'None', 1, 'None', 1, 'None', 1, 'None', 1]
#         lora_config_index = 12
#         values = config[:lora_config_index] + lora_config + config[lora_config_index:]


#         print("Attempting to gen image with values: ", values)

#         # need to call set_all_advanced_parameters before running a task
#         advanced_parameters.set_all_advanced_parameters(False, 1.5, 0.8, 0.3, 7, 'dpmpp_2m_sde_gpu', 'karras', False, -1, -1, -1, -1, -1, -1, False, False, False, False, 0.25, 64, 128, 'joint', False, 1.01, 1.02, 0.99, 0.95, False, False, 'v2.6', 1, 0.618)
#         # task = worker.AsyncTask(args=params)
#         task = worker.AsyncTask(args=values)

#         worker.async_tasks.append(task)

#         print('taks queued')

#         finished = False
#         while not finished:
#             time.sleep(0.01)
#             if len(task.yields) > 0:
#                 flag, product = task.yields.pop(0)
#                 if flag == 'preview':

#                     # help bad internet connection by skipping duplicated preview
#                     if len(task.yields) > 0:  # if we have the next item
#                         if task.yields[0][0] == 'preview':   # if the next item is also a preview
#                             # print('Skipped one preview for better internet connection.')
#                             continue

#                     percentage, title, image = product
#                     print ('NOT FINISHED, preview', percentage, title)
#                     # yield (percentage, title)
#                     if isinstance(image, np.ndarray):
#                         print("Image shape: ", image.shape)
#                         image = image.tolist()
#                     data = { 
#                         "updateType": "preview", 
#                         "percentage": percentage, 
#                         "title": title, 
#                         "image": image
#                         }
#                     json_data = json.dumps(data)
#                     # yield f"{json_data}\n\n"

#                 if flag == 'results':
#                     print ('RESULTS, results')
#                     print(f"The type of 'product' is: {type(product)}")
#                     if isinstance(product, np.ndarray):
#                         print("Product shape: ", product.shape)
#                         product = product.tolist()
#                     elif isinstance(product, list):
#                         # Convert each ndarray element in the list to a list
#                         product = [item.tolist() if isinstance(item, np.ndarray) else item for item in product]
#                     else:
#                         product = product
#                     data = {
#                         "updateType": "results", 
#                         "product": product
#                     }
#                     json_data = json.dumps(data)
#                     # yield f"{json_data}\n\n"

#                 if flag == 'finish':
#                     print ('FINISHED')
#                     print(f"The type of 'product' is: {type(product)}")
#                     if isinstance(product, np.ndarray):
#                         print("Product shape: ", product.shape)
#                         product = product.tolist()
#                     elif isinstance(product, list):
#                         # Convert each ndarray element in the list to a list
#                         product = [item.tolist() if isinstance(item, np.ndarray) else item for item in product]
#                     else:
#                         product = product
#                     data = {
#                         "updateType": "finished", 
#                         "product": product
#                     }
#                     return jsonify(product)

#                     finished = True


#         # print("---image/gen endpoint results: ", task.results)

#         # return Response(response_stream(task), mimetype='text/event-stream')

#         # return jsonify("Horrya")
#     except ValidationError as err:
#         return jsonify(err.messages), 400
    
#     return jsonify({"message": "Something unexpected happened"}), 500

