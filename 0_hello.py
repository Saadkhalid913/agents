import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)


def main():
    try:
        completion = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello world"}
            ]
        )
        print(completion.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
