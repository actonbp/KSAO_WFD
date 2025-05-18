#!/usr/bin/env python3
"""
Simplified test script for the Gemini API.
"""

import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Get the API key
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not found. Please set it in your .env file.")

# Initialize the Gemini API client
genai_client = genai.Client(api_key=api_key)

# Simple test with a short prompt
try:
    print("Sending a simple test request to Gemini API...")
    
    prompt = "List 5 knowledge, skills, abilities, and other characteristics (KSAOs) that a substance abuse counselor should have, organized by category:"
    
    response = genai_client.models.generate_content(
        model="gemini-2.5-pro-preview-05-06",
        contents=prompt
    )
    
    print("\nResponse from Gemini API:")
    print(response.text)
    
    print("\nTest completed successfully!")
    
except Exception as e:
    print(f"Error during API call: {e}")