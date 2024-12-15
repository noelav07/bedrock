import boto3
import json

# Initialize the Bedrock client with the correct region
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-2")

def get_user_input():
    """
    Prompt the user for input.
    """
    prompt_data = input("Enter the prompt for the AI model: ")
    return prompt_data

# Get dynamic input from the user
prompt_data = get_user_input()

# Prepare the payload
payload = {
    "prompt": "[INST] " + prompt_data + " [/INST]",
    "max_gen_len": 512,
    "temperature": 0.5,
    "top_p": 0.9
}

# Convert the payload to JSON
body = json.dumps(payload)

# Model ID
model_id = "us.meta.llama3-2-90b-instruct-v1:0"

try:
    # Invoke the model with the corrected model ID and parameters
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,            # Correct parameter name
        accept="application/json",   # Correct parameter name
        contentType="application/json"  # Correct parameter name
    )

    # Read and parse the response
    response_body = json.loads(response['body'].read())

    # Extract the generated text (check for potential keys)
    response_text = response_body.get('generation') or response_body.get('outputs') or response_body.get('completions', 'No generation returned')

    print("\nGenerated Response:")
    print(response_text)

except Exception as e:
    print(f"Error occurred: {e}")
