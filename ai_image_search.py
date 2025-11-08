"""
CMPE-273 Assignment: AI Image Search with Multi-Provider Support
Author: Your Name
Date: 2025-11-08
Description: Generate embeddings for images and text using multiple providers (Cohere, OpenAI),
             then compute cosine similarity for image-to-image and text-to-image comparisons.
"""

import numpy as np
import requests
import base64
from typing import List, Tuple

# Optional imports - will be loaded based on API_PROVIDER selection
try:
    import cohere
except ImportError:
    cohere = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


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


def generate_embeddings_cohere(api_key: str, images: List[str], texts: List[str]) -> Tuple[List, List]:
    """Generate embeddings using Cohere API."""
    if cohere is None:
        raise ImportError("Cohere SDK not installed. Run: pip install cohere")

    co = cohere.ClientV2(api_key=api_key)

    # Generate image embeddings
    print("  Using Cohere embed-v4.0 model...")
    image_response = co.embed(
        images=images,
        model="embed-v4.0",
        input_type="image",
        embedding_types=["float"]
    )
    image_embeddings = image_response.embeddings.float

    # Generate text embeddings
    text_response = co.embed(
        texts=texts,
        model="embed-v4.0",
        input_type="search_query",
        embedding_types=["float"]
    )
    text_embeddings = text_response.embeddings.float

    return image_embeddings, text_embeddings


def generate_embeddings_openai(api_key: str, images: List[str], texts: List[str], image_urls: List[str]) -> Tuple[List, List]:
    """Generate embeddings using OpenAI API."""
    if OpenAI is None:
        raise ImportError("OpenAI SDK not installed. Run: pip install openai")

    client = OpenAI(api_key=api_key)

    print("  Using OpenAI text-embedding-3-large model...")
    print("  Note: OpenAI doesn't support direct image embeddings via embeddings API.")
    print("  Using image URLs as text for demonstration purposes.")

    # For images, we'll embed the image URLs as text (limitation of OpenAI embeddings API)
    # OpenAI's embeddings API doesn't support images directly
    image_embeddings = []
    for url in image_urls:
        response = client.embeddings.create(
            input=f"Image from URL: {url}",
            model="text-embedding-3-large"
        )
        image_embeddings.append(response.data[0].embedding)

    # Generate text embeddings
    text_embeddings = []
    for text in texts:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-large"
        )
        text_embeddings.append(response.data[0].embedding)

    return image_embeddings, text_embeddings


def main():
    """Main function to execute the AI Image Search assignment."""

    # ==================== CONFIGURATION ====================
    # Choose your API provider: "cohere" or "openai"
    API_PROVIDER = "cohere"  # Change to "openai" to use OpenAI instead

    # API Keys - Replace with your actual key
    COHERE_API_KEY = "YOUR_COHERE_API_KEY_HERE"
    OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_HERE"
    # =======================================================

    print("=" * 80)
    print(f"CMPE-273: AI Image Search with {API_PROVIDER.upper()} API")
    print("=" * 80)
    print()

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

    # Step 2 & 3: Generate embeddings based on selected provider
    print(f"Step 2 & 3: Generating embeddings using {API_PROVIDER.upper()} API...")
    print("-" * 80)
    try:
        if API_PROVIDER.lower() == "cohere":
            api_key = COHERE_API_KEY
            if api_key == "YOUR_COHERE_API_KEY_HERE":
                print("  ✗ Error: Please set your Cohere API key in the COHERE_API_KEY variable")
                return
            image_embeddings, text_embeddings = generate_embeddings_cohere(
                api_key, encoded_images, text_queries
            )

        elif API_PROVIDER.lower() == "openai":
            api_key = OPENAI_API_KEY
            if api_key == "YOUR_OPENAI_API_KEY_HERE":
                print("  ✗ Error: Please set your OpenAI API key in the OPENAI_API_KEY variable")
                return
            image_embeddings, text_embeddings = generate_embeddings_openai(
                api_key, encoded_images, text_queries, image_urls
            )

        else:
            print(f"  ✗ Error: Unsupported API provider: {API_PROVIDER}")
            print("  Supported providers: 'cohere', 'openai'")
            return

        print(f"  ✓ Generated embeddings for {len(image_embeddings)} images")
        print(f"  ✓ Embedding dimension: {len(image_embeddings[0])}")
        print(f"  ✓ Generated embeddings for {len(text_embeddings)} text queries")

    except Exception as e:
        print(f"  ✗ Error generating embeddings: {e}")
        import traceback
        traceback.print_exc()
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

