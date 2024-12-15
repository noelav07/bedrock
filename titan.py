import base64
import io
import json
import logging
import boto3
from PIL import Image
from botocore.exceptions import ClientError


class ImageError(Exception):
    """Custom exception for errors returned by Amazon Titan Image Generator G1"""

    def __init__(self, message):
        self.message = message


# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def generate_image(model_id, body):
    """
    Generate an image using Amazon Titan Image Generator G1 model on demand.

    Args:
        model_id (str): The model ID to use.
        body (str): The request body to use.

    Returns:
        image_bytes (bytes): The image generated by the model.
    """
    logger.info("Generating image with Amazon Titan Image Generator G1 model %s", model_id)

    bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

    accept = "application/json"
    content_type = "application/json"

    try:
        response = bedrock.invoke_model(
            body=body, modelId=model_id, accept=accept, contentType=content_type
        )
        response_body = json.loads(response.get("body").read())
    except ClientError as err:
        raise ImageError(f"ClientError: {err.response['Error']['Message']}")

    base64_image = response_body.get("images")[0]
    base64_bytes = base64_image.encode('ascii')
    image_bytes = base64.b64decode(base64_bytes)

    finish_reason = response_body.get("error")

    if finish_reason is not None:
        raise ImageError(f"Image generation error: {finish_reason}")

    logger.info("Successfully generated image with Amazon Titan Image Generator G1 model %s", model_id)

    return image_bytes


def main():
    """
    Entrypoint for Amazon Titan Image Generator G1 example.
    """
    # Prompt the user for a description of the image
    prompt = input("Enter a description for the image to generate: ")

    # Define the model ID for the Titan Image Generator
    model_id = 'amazon.titan-image-generator-v1'

    # Prepare the request payload
    body = json.dumps({
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "height": 1024,
            "width": 1024,
            "cfgScale": 8.0,
            "seed": 0
        }
    })

    try:
        # Generate the image
        image_bytes = generate_image(model_id=model_id, body=body)
        
        # Save the image to a file
        output_file = "generated_image2.png"
        with open(output_file, "wb") as f:
            f.write(image_bytes)

        print(f"Image successfully generated and saved as '{output_file}'.")
        logger.info(f"Image saved to '{output_file}'.")

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print(f"A client error occurred: {message}")
    except ImageError as err:
        logger.error(err.message)
        print(err.message)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred: {e}")

    else:
        print(f"Finished generating image with Amazon Titan Image Generator G1 model {model_id}.")


if __name__ == "__main__":
    main()
