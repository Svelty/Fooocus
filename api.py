from flask import Flask, request, jsonify, Response
from marshmallow import Schema, fields, ValidationError, post_load, validate
import modules.async_worker as worker
import modules.advanced_parameters as advanced_parameters
import time
import json
import sys
import numpy as np

from flask_cors import CORS

#gross
print('[System ARGV] ' + str(sys.argv))
temp = sys.argv
sys.argv = ['launch_from_api.py', '--preset', 'anime']
from launch_from_api import *
sys.argv = temp
#

app = Flask(__name__)
CORS(app)

#TODO: instead of doing this i should start this api from the command line then import a version of launch.py that doesn't include the webUI
# def run_app():
#     app.run(debug=True, use_reloader=False)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route('/json', methods=['POST'])
def json_post():
    # json keys must be strings
    return {
        "this": "is",
        "json": "kindof"
    }


class ImageGenSchema(Schema):
    # name = fields.Str(required=True, validate=lambda s: 4 <= len(s) <= 25)
    # age = fields.Int(required=True, validate=lambda n: 0 <= n <= 100)

    # loras_parameters = fields.List(fields.List(fields.Raw()), required=False, data_key="lorasParameters")
    refiner_switch = fields.Float(required=False, data_key="refinerSwitch")
    refiner_model_name = fields.Str(required=False, data_key="refinerModelName")
    base_model_name = fields.Str(required=True, data_key="baseModelName")
    guidance_scale = fields.Int(required=False, data_key="guidanceScale")
    sharpness = fields.Int(required=False, data_key="sharpness")
    image_seed = fields.Str(required=True, data_key="imageSeed")
    image_number = fields.Int(required=False, data_key="imageNumber")
    aspect_ratios_selection = fields.Str(required=True, data_key="aspectRatiosSelection")
    performance_selection = fields.Str(
        required=True, 
        data_key="performanceSelection", 
        validate=validate.OneOf(["Speed", "Quality", "Extreme Speed"])
    )
    style_selections = fields.List(fields.Str(), required=False, data_key="styleSelections")
    negative_prompt = fields.Str(required=False, data_key="negativePrompt")
    prompt = fields.Str(required=True, data_key="prompt")

    @post_load
    def make_snake_case(self, data, **kwargs):
        # Convert camelCase keys to snake_case
        return {self.camel_to_snake(key): value for key, value in data.items()}

    @staticmethod
    def camel_to_snake(name):
        import re
        str1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', str1).lower()

@app.route('/image/gen', methods=['POST'])
def process_params():
    print("Image gen endpoint")

    schema = ImageGenSchema()
    try:
        data = schema.load(request.json)
        
        config = [
            data.get('prompt'),
            data.get('negative_prompt') if data.get('negative_prompt') else '',
            data.get('style_selections') if data.get('style_selections') else [], 
            data.get('performance_selection'),
            data.get('aspect_ratios_selection'), #'1152×896', #TODO: aspect_ratio '1152×896 <span style="color: grey;"> ∣ 9:7</span>',
            data.get('image_number') if data.get('image_number') else 1,
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

        lora_config = ['sd_xl_offset_example-lora_1.0.safetensors', 0.5, 'None', 1, 'None', 1, 'None', 1, 'None', 1]
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

        return Response(response_stream(task), mimetype='text/event-stream')

        return jsonify("Horrya")
    except ValidationError as err:
        return jsonify(err.messages), 400
    
    return jsonify({"message": "Something unexpected happened"}), 500

def response_stream(task):
    data = { "updateType":"init", "percentage": 1, "title": 'Waiting for task to start ...', }
    json_data = json.dumps(data)
    yield f"{json_data}\n\n"

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
                if isinstance(image, np.ndarray):
                    print("Image shape: ", image.shape)
                    image = image.tolist()
                data = { 
                    "updateType": "preview", 
                    "percentage": percentage, 
                    "title": title, 
                    "image": image
                    }
                json_data = json.dumps(data)
                yield f"{json_data}\n\n"
                # yield gr.update(visible=True, value=modules.html.make_progress_html(percentage, title)), \
                #     gr.update(visible=True, value=image) if image is not None else gr.update(), \
                #     gr.update(), \
                #     gr.update(visible=False)
            if flag == 'results':
                print ('RESULTS, results')
                print(f"The type of 'product' is: {type(product)}")
                if isinstance(product, np.ndarray):
                    print("Product shape: ", product.shape)
                    product = product.tolist()
                elif isinstance(product, list):
                    # Convert each ndarray element in the list to a list
                    product = [item.tolist() if isinstance(item, np.ndarray) else item for item in product]
                else:
                    product = product
                data = {
                    "updateType": "results", 
                    "product": product
                }
                json_data = json.dumps(data)
                yield f"{json_data}\n\n"
                # yield gr.update(visible=True), \
                #     gr.update(visible=True), \
                #     gr.update(visible=True, value=product), \
                #     gr.update(visible=False)
            if flag == 'finish':
                print ('FINISHED')
                print(f"The type of 'product' is: {type(product)}")
                if isinstance(product, np.ndarray):
                    print("Product shape: ", product.shape)
                    product = product.tolist()
                elif isinstance(product, list):
                    # Convert each ndarray element in the list to a list
                    product = [item.tolist() if isinstance(item, np.ndarray) else item for item in product]
                else:
                    product = product
                data = {
                    "updateType": "finished", 
                    "product": product
                }
                json_data = json.dumps(data)
                yield f"{json_data}\n\n"
                # yield gr.update(visible=False), \
                #     gr.update(visible=False), \
                #     gr.update(visible=False), \
                #     gr.update(visible=True, value=product)
                # print("TASK FINISHED, results:", task.results)
                # print("TASK FINISHED, product:", product)
                finished = True




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

# def response_stream(task):
#     data = { "updateType":"init", "percentage": 1, "title": 'Waiting for task to start ...', }
#     json_data = json.dumps(data)
#     yield f"{json_data}\n\n"

#     finished = False
#     while not finished:
#         time.sleep(0.01)
#         if len(task.yields) > 0:
#             flag, product = task.yields.pop(0)
#             if flag == 'preview':

#                 # help bad internet connection by skipping duplicated preview
#                 if len(task.yields) > 0:  # if we have the next item
#                     if task.yields[0][0] == 'preview':   # if the next item is also a preview
#                         # print('Skipped one preview for better internet connection.')
#                         continue

#                 percentage, title, image = product
#                 print ('NOT FINISHED, preview', percentage, title)
#                 # yield (percentage, title)
#                 if isinstance(image, np.ndarray):
#                     print("Image shape: ", image.shape)
#                     image = image.tolist()
#                 data = { 
#                     "updateType": "preview", 
#                     "percentage": percentage, 
#                     "title": title, 
#                     "image": image
#                     }
#                 json_data = json.dumps(data)
#                 yield f"{json_data}\n\n"
#                 # yield gr.update(visible=True, value=modules.html.make_progress_html(percentage, title)), \
#                 #     gr.update(visible=True, value=image) if image is not None else gr.update(), \
#                 #     gr.update(), \
#                 #     gr.update(visible=False)
#             if flag == 'results':
#                 print ('RESULTS, results')
#                 print(f"The type of 'product' is: {type(product)}")
#                 if isinstance(product, np.ndarray):
#                     print("Product shape: ", product.shape)
#                     product = product.tolist()
#                 elif isinstance(product, list):
#                     # Convert each ndarray element in the list to a list
#                     product = [item.tolist() if isinstance(item, np.ndarray) else item for item in product]
#                 else:
#                     product = product
#                 data = {
#                     "updateType": "results", 
#                     "product": product
#                 }
#                 json_data = json.dumps(data)
#                 yield f"{json_data}\n\n"
#                 # yield gr.update(visible=True), \
#                 #     gr.update(visible=True), \
#                 #     gr.update(visible=True, value=product), \
#                 #     gr.update(visible=False)
#             if flag == 'finish':
#                 print ('FINISHED')
#                 print(f"The type of 'product' is: {type(product)}")
#                 if isinstance(product, np.ndarray):
#                     print("Product shape: ", product.shape)
#                     product = product.tolist()
#                 elif isinstance(product, list):
#                     # Convert each ndarray element in the list to a list
#                     product = [item.tolist() if isinstance(item, np.ndarray) else item for item in product]
#                 else:
#                     product = product
#                 data = {
#                     "updateType": "finished", 
#                     "product": product
#                 }
#                 json_data = json.dumps(data)
#                 yield f"{json_data}\n\n"
#                 # yield gr.update(visible=False), \
#                 #     gr.update(visible=False), \
#                 #     gr.update(visible=False), \
#                 #     gr.update(visible=True, value=product)
#                 # print("TASK FINISHED, results:", task.results)
#                 # print("TASK FINISHED, product:", product)
#                 finished = True