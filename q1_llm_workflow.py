from google import genai 

client = genai.Client(
    api_key="AQ.Ab8RN6KjqSW6dV56BW2aMsfa8dE8Yp8J9v1x7ooqUeUsqF1KOg"
)

question = input("Enter your question: ") 

# Call Gemini model
response = client.models.generate_content(
    model="gemini-3.5-flash", 
    contents=question
)

print("\nResponse:") 
print(response.text)
