import base64
import json

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_payload(image_path, food_type, plate_diameter="0.7"):
    encoded_image = encode_image(image_path)
    payload = {
        "img": encoded_image,
        "food_type": food_type,
        "plate_diameter": plate_diameter
    }
    with open("payload.json", "w") as json_file:
        json.dump(payload, json_file)

create_payload("apple.jpg", "apple")
