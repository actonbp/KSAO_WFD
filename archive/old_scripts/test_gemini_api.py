import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Get the API key from the environment variable
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY environment variable not found in .env or environment.")
    print("Please ensure it is set in your .env file or exported in your shell.")
else:
    try:
        # Initialize the client with your API key
        client = genai.Client(api_key=api_key)
        
        # Use the latest Gemini 2.5 Pro Preview model
        model_name = "gemini-2.5-pro-preview-05-06"
        print(f"Using model: {model_name}")
        
        # Prepare the prompt
        prompt = "Explain how AI works in a few words"
        
        # Generate content using the model
        response = client.models.generate_content(
            model="gemini-2.5-pro-preview-05-06",
            contents=prompt
        )
        
        print(f"Prompt: {prompt}")
        print(f"Response: {response.text}")
        
        print("\nTest successful! Gemini 2.5 Pro Preview is working correctly.")

    except AttributeError as ae:
        print(f"An AttributeError occurred: {ae}")
        print("This might indicate an issue with how the library is imported or used.")
        print("Attempted import: from google import genai")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your API key is correct and has the necessary permissions.")
        print("You might also want to check if you have access to the specified model.") 