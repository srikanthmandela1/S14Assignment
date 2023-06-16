# S14Assignment

pip install openai
openai.api_key = 'YOUR_API_KEY'

import openai
from openai.error import OpenAIError

# Set up OpenAI API credentials
openai.api_key = 'YOUR_API_KEY'

# Define the prompt and the different styles
prompt = "What is the capital of France?"
styles = [
    {
        "name": "Formal",
        "seed": 12345,
    },
    {
        "name": "Casual",
        "seed": 98765,
    },
    {
        "name": "Poetic",
        "seed": 24680,
    },
    {
        "name": "Technical",
        "seed": 13579,
    },
    {
        "name": "Playful",
        "seed": 54321,
    },
]

# Generate outputs for each style
for style in styles:
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=50,
            temperature=0.6,
            n=1,
            stop=None,
            seed=style["seed"],
        )
        output = response.choices[0].text.strip()

        print(f"Style: {style['name']}")
        print(f"Output: {output}")
        print()
    except OpenAIError as e:
        print(f"OpenAI Error: {e}")


import numpy as np

# Define a custom loss function
def blue_loss(predictions, targets):
    # Calculate the difference between predictions and targets
    diff = predictions - targets

    # Apply any desired transformation or calculation on the difference
    # For example, you can square the difference and take the mean
    loss = np.mean(np.square(diff))

    return loss

# Example usage
predictions = np.array([0.5, 0.7, 0.3])
targets = np.array([0.8, 0.9, 0.2])

loss = blue_loss(predictions, targets)
print(f"Blue Loss: {loss}")

pip install tweepy



import tweepy

# Twitter API credentials
consumer_key = 'YOUR_CONSUMER_KEY'
consumer_secret = 'YOUR_CONSUMER_SECRET'
access_token = 'YOUR_ACCESS_TOKEN'
access_token_secret = 'YOUR_ACCESS_TOKEN_SECRET'

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Create API object
api = tweepy.API(auth)

# List of image file paths
image_files = ['image1.jpg', 'image2.jpg', 'image3.jpg']

# Upload images and post tweets
for image_file in image_files:
    try:
        # Upload image
        media = api.media_upload(image_file)

        # Post tweet with the uploaded image
        tweet_text = 'Check out this amazing image!'
        post_tweet = api.update_status(status=tweet_text, media_ids=[media.media_id])
        print(f"Tweet posted with image: {image_file}")
    except Exception as e:
        print(f"Error posting tweet with image {image_file}: {e}")


