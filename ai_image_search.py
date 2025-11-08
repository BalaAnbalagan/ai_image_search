"""
CMPE-273 Assignment: AI Image Search with Cohere Embeddings
Author: Your Name
Date: 2025-11-08
Description: Generate embeddings for images and text using Cohere embed-v4.0,
             then compute cosine similarity for image-to-image and text-to-image comparisons.
"""

import cohere
import numpy as np
import requests
import base64


def download_and_encode_image(url):
    """
    Download an image from URL and encode it to base64.

    Args:
        url (str): URL of the image

    Returns:
        str: Base64 encoded image
    """
    print(f"  Downloading: {url}")
    response = requests.get(url)
    response.raise_for_status()

    # Encode image to base64
    image_bytes = response.content
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    return base64_image


def compute_cosine_similarity(embedding1, embedding2):
    """
    Compute cosine similarity between two embeddings.

    Args:
        embedding1 (list): First embedding vector
        embedding2 (list): Second embedding vector

    Returns:
        float: Cosine similarity score
    """
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)

    # Cosine similarity formula: (A · B) / (||A|| * ||B||)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    similarity = dot_product / (norm1 * norm2)

    return similarity


def main():
    """Main function to execute the AI Image Search assignment."""

    print("=" * 80)
    print("CMPE-273: AI Image Search with Cohere Embeddings (embed-v4.0)")
    print("=" * 80)
    print()

    # Initialize Cohere client
    API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual Cohere API key
    co = cohere.ClientV2(api_key=API_KEY)

    # Define image URLs
    image_urls = [
        "https://www.sjsu.edu/_images/people/ADV_college-of-science_2.jpg",
        "https://www.sjsu.edu/_images/people/ADV_college-of-social-sciences_2.jpg"
    ]

    # Define text queries
    text_queries = [
        "person with tape and cap",
        "cart with single tire"
    ]

    # Step 1: Download and encode images
    print("Step 1: Downloading and encoding images...")
    print("-" * 80)
    encoded_images = []
    for idx, url in enumerate(image_urls, 1):
        try:
            encoded_img = download_and_encode_image(url)
            encoded_images.append(encoded_img)
            print(f"  ✓ Image {idx} encoded successfully")
        except Exception as e:
            print(f"  ✗ Error downloading image {idx}: {e}")
            return
    print()

    # Step 2: Generate image embeddings
    print("Step 2: Generating image embeddings using Cohere embed-v4.0...")
    print("-" * 80)
    try:
        image_response = co.embed(
            images=encoded_images,
            model="embed-v4.0",
            input_type="image",
            embedding_types=["float"]
        )
        image_embeddings = image_response.embeddings.float
        print(f"  ✓ Generated embeddings for {len(image_embeddings)} images")
        print(f"  ✓ Embedding dimension: {len(image_embeddings[0])}")
    except Exception as e:
        print(f"  ✗ Error generating image embeddings: {e}")
        return
    print()

    # Step 3: Generate text embeddings
    print("Step 3: Generating text embeddings using Cohere embed-v4.0...")
    print("-" * 80)
    try:
        text_response = co.embed(
            texts=text_queries,
            model="embed-v4.0",
            input_type="search_query",
            embedding_types=["float"]
        )
        text_embeddings = text_response.embeddings.float
        print(f"  ✓ Generated embeddings for {len(text_embeddings)} text queries")
        print(f"  ✓ Embedding dimension: {len(text_embeddings[0])}")
    except Exception as e:
        print(f"  ✗ Error generating text embeddings: {e}")
        return
    print()

    # Step 4: Compute image-to-image similarity
    print("Step 4: Computing Image-to-Image Cosine Similarity")
    print("=" * 80)
    img_to_img_similarity = compute_cosine_similarity(
        image_embeddings[0],
        image_embeddings[1]
    )
    print(f"Image 1 (College of Science) vs Image 2 (College of Social Sciences):")
    print(f"  Cosine Similarity: {img_to_img_similarity:.6f}")
    print()

    # Step 5: Compute text-to-image similarities
    print("Step 5: Computing Text-to-Image Cosine Similarities")
    print("=" * 80)

    for text_idx, text_query in enumerate(text_queries):
        print(f"\nText Query: \"{text_query}\"")
        print("-" * 80)

        for img_idx in range(len(image_embeddings)):
            similarity = compute_cosine_similarity(
                text_embeddings[text_idx],
                image_embeddings[img_idx]
            )
            image_name = "College of Science" if img_idx == 0 else "College of Social Sciences"
            print(f"  vs Image {img_idx + 1} ({image_name}):")
            print(f"    Cosine Similarity: {similarity:.6f}")

    print()
    print("=" * 80)
    print("Summary of Results")
    print("=" * 80)
    print(f"\n1. Image-to-Image Similarity:")
    print(f"   • Both SJSU college images: {img_to_img_similarity:.6f}")
    print(f"\n2. Text-to-Image Similarities:")

    for text_idx, text_query in enumerate(text_queries):
        print(f"\n   Query: \"{text_query}\"")
        for img_idx in range(len(image_embeddings)):
            similarity = compute_cosine_similarity(
                text_embeddings[text_idx],
                image_embeddings[img_idx]
            )
            image_name = "Science" if img_idx == 0 else "Social Sciences"
            print(f"     • {image_name}: {similarity:.6f}")

    print()
    print("=" * 80)
    print("Assignment completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

