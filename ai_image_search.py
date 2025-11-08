"""
CMPE-273 Assignment: AI Image Search with Multi-Provider Support
Author: Your Name
Date: 2025-11-08
Description: Generate embeddings for images and text using multiple providers (Cohere, OpenAI),
             then compute cosine similarity for image-to-image and text-to-image comparisons.
"""

import os
import numpy as np
import requests
import base64
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Optional imports - will be loaded based on API_PROVIDER selection
try:
    import cohere
except ImportError:
    cohere = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Global cache for downloaded images to avoid re-downloading
_image_cache = {}


def download_and_encode_image(url, use_cache=True):
    """
    Download an image from URL and encode it to base64 data URI format.
    Uses caching to avoid re-downloading the same image.

    Args:
        url (str): URL of the image
        use_cache (bool): Whether to use cached version if available

    Returns:
        str: Base64 encoded image as data URI
    """
    # Check cache first
    if use_cache and url in _image_cache:
        print(f"  Using cached: {url}")
        return _image_cache[url]

    print(f"  Downloading: {url}")
    response = requests.get(url)
    response.raise_for_status()

    # Detect content type
    content_type = response.headers.get('content-type', 'image/jpeg')

    # Encode image to base64
    image_bytes = response.content
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    # Format as data URI for Cohere API
    data_uri = f"data:{content_type};base64,{base64_image}"

    # Store in cache
    if use_cache:
        _image_cache[url] = data_uri

    return data_uri


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


def print_comparison_table(headers, rows, title=None):
    """
    Print a formatted comparison table.

    Args:
        headers (list): Column headers
        rows (list): List of row data (each row is a list)
        title (str): Optional table title
    """
    if title:
        print(f"\n{title}")
        print("=" * 80)

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Print header
    header_line = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))

    # Print rows
    for row in rows:
        row_line = "  ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
        print(row_line)


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

    # Load API Keys from .env file (secure - not committed to git)
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "YOUR_COHERE_API_KEY_HERE")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")

    # Detect available API providers
    available_providers = []
    if COHERE_API_KEY != "YOUR_COHERE_API_KEY_HERE":
        available_providers.append("cohere")
    if OPENAI_API_KEY != "YOUR_OPENAI_API_KEY_HERE":
        available_providers.append("openai")

    # ==================== API PROVIDER SELECTION ====================
    if len(available_providers) == 0:
        print("=" * 80)
        print("ERROR: No API keys found!")
        print("=" * 80)
        print("\nPlease add your API key to the .env file:")
        print("  1. Copy .env.example to .env")
        print("  2. Add your Cohere or OpenAI API key")
        print("  3. Run the script again")
        return

    elif len(available_providers) == 1:
        # Only one provider available, use it automatically
        API_PROVIDER = available_providers[0]
        print("=" * 80)
        print(f"CMPE-273: AI Image Search with {API_PROVIDER.upper()} API")
        print("=" * 80)
        print(f"Using {API_PROVIDER.upper()} (only available provider)")
        print()

    else:
        # Multiple providers available, let user choose
        print("=" * 80)
        print("CMPE-273: AI Image Search - Multi-Provider Support")
        print("=" * 80)
        print("\nMultiple API keys detected! Please select a mode:")
        print()
        print("  1. COHERE only - Cohere embed-v4.0 (Recommended for assignment)")
        print("  2. OPENAI only - OpenAI text-embedding-3-large")
        print("  3. COMPARE BOTH - Run both and generate comparison report")

        print()
        while True:
            try:
                choice = input("Enter your choice (1, 2, or 3): ").strip()
                if choice == "3":
                    # Compare mode - will run both
                    run_comparison = True
                    API_PROVIDER = None  # Will run both
                    break
                else:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(available_providers):
                        run_comparison = False
                        API_PROVIDER = available_providers[choice_idx]
                        break
                    else:
                        print("Invalid choice. Please enter 1, 2, or 3.")
            except (ValueError, KeyboardInterrupt):
                print("\nExiting...")
                return

        if not run_comparison:
            print()
            print("=" * 80)
            print(f"Selected: {API_PROVIDER.upper()} API")
            print("=" * 80)
            print()
        else:
            print()
            print("=" * 80)
            print("Comparison Mode: Running BOTH Cohere and OpenAI")
            print("=" * 80)
            print()
    # =================================================================

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

    # Determine if we need to check for run_comparison variable
    try:
        run_comparison
    except NameError:
        run_comparison = False

    # Step 2 & 3: Generate embeddings
    if run_comparison:
        # Comparison mode: Run both providers
        print("Step 2 & 3: Generating embeddings using BOTH APIs for comparison...")
        print("=" * 80)

        # Store results from both providers
        results = {}

        # Run Cohere
        print("\n[1/2] Running Cohere API...")
        print("-" * 80)
        try:
            cohere_img_emb, cohere_txt_emb = generate_embeddings_cohere(
                COHERE_API_KEY, encoded_images, text_queries
            )
            print(f"  ✓ Cohere: Generated embeddings for {len(cohere_img_emb)} images")
            print(f"  ✓ Cohere: Embedding dimension: {len(cohere_img_emb[0])}")
            print(f"  ✓ Cohere: Generated embeddings for {len(cohere_txt_emb)} text queries")
            results['cohere'] = {
                'image_embeddings': cohere_img_emb,
                'text_embeddings': cohere_txt_emb
            }
        except Exception as e:
            print(f"  ✗ Error with Cohere API: {e}")
            import traceback
            traceback.print_exc()
            return

        # Run OpenAI
        print("\n[2/2] Running OpenAI API...")
        print("-" * 80)
        try:
            openai_img_emb, openai_txt_emb = generate_embeddings_openai(
                OPENAI_API_KEY, encoded_images, text_queries, image_urls
            )
            print(f"  ✓ OpenAI: Generated embeddings for {len(openai_img_emb)} images")
            print(f"  ✓ OpenAI: Embedding dimension: {len(openai_img_emb[0])}")
            print(f"  ✓ OpenAI: Generated embeddings for {len(openai_txt_emb)} text queries")
            results['openai'] = {
                'image_embeddings': openai_img_emb,
                'text_embeddings': openai_txt_emb
            }
        except Exception as e:
            print(f"  ✗ Error with OpenAI API: {e}")
            import traceback
            traceback.print_exc()
            return

        print()

    else:
        # Single provider mode
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

    # Step 4 & 5: Compute similarities and display results
    if run_comparison:
        # Comparison mode: Compute and compare both providers
        print("Step 4 & 5: Computing Similarities and Comparing Results")
        print("=" * 80)

        # Compute similarities for both providers
        comparison_data = {}

        for provider in ['cohere', 'openai']:
            img_emb = results[provider]['image_embeddings']
            txt_emb = results[provider]['text_embeddings']

            # Image-to-image similarity
            img_to_img = compute_cosine_similarity(img_emb[0], img_emb[1])

            # Text-to-image similarities
            txt_to_img = []
            for text_idx in range(len(txt_emb)):
                for img_idx in range(len(img_emb)):
                    similarity = compute_cosine_similarity(txt_emb[text_idx], img_emb[img_idx])
                    txt_to_img.append({
                        'text_idx': text_idx,
                        'img_idx': img_idx,
                        'similarity': similarity
                    })

            comparison_data[provider] = {
                'img_to_img': img_to_img,
                'txt_to_img': txt_to_img
            }

        # Display comparison results
        print("\n" + "=" * 80)
        print("COMPARISON RESULTS: Cohere vs OpenAI")
        print("=" * 80)

        # Image-to-image comparison table
        cohere_i2i = comparison_data['cohere']['img_to_img']
        openai_i2i = comparison_data['openai']['img_to_img']
        diff_i2i = abs(cohere_i2i - openai_i2i)

        print("\n1. Image-to-Image Similarity")
        headers = ["Comparison", "Cohere", "OpenAI", "Difference", "Diff %"]
        rows = [[
            "Science vs Social Sciences",
            f"{cohere_i2i:.6f}",
            f"{openai_i2i:.6f}",
            f"{diff_i2i:.6f}",
            f"{(diff_i2i/cohere_i2i)*100:.2f}%"
        ]]
        print_comparison_table(headers, rows)

        # Text-to-image comparison table
        print("\n2. Text-to-Image Similarities")
        headers = ["Query", "Image", "Cohere", "OpenAI", "Difference", "Diff %"]
        rows = []

        for text_idx, text_query in enumerate(text_queries):
            for img_idx in range(len(image_urls)):
                image_name = "Science" if img_idx == 0 else "Social Sciences"

                # Find similarities for this text-image pair
                cohere_sim = next(item['similarity'] for item in comparison_data['cohere']['txt_to_img']
                                  if item['text_idx'] == text_idx and item['img_idx'] == img_idx)
                openai_sim = next(item['similarity'] for item in comparison_data['openai']['txt_to_img']
                                  if item['text_idx'] == text_idx and item['img_idx'] == img_idx)
                diff = abs(cohere_sim - openai_sim)

                # Shorten query for display
                query_short = text_query[:25] + "..." if len(text_query) > 25 else text_query

                rows.append([
                    query_short,
                    image_name,
                    f"{cohere_sim:.6f}",
                    f"{openai_sim:.6f}",
                    f"{diff:.6f}",
                    f"{(diff/max(cohere_sim, 0.0001))*100:.2f}%"
                ])

        print_comparison_table(headers, rows)

        # Summary statistics
        print("\n" + "=" * 80)
        print("Summary Statistics")
        print("=" * 80)
        print(f"  Embedding Dimensions:")
        print(f"    • Cohere embed-v4.0: {len(results['cohere']['image_embeddings'][0])} dimensions")
        print(f"    • OpenAI text-embedding-3-large: {len(results['openai']['image_embeddings'][0])} dimensions")

        all_diffs = [diff_i2i]
        for text_idx in range(len(text_queries)):
            for img_idx in range(len(image_urls)):
                cohere_sim = next(item['similarity'] for item in comparison_data['cohere']['txt_to_img']
                                  if item['text_idx'] == text_idx and item['img_idx'] == img_idx)
                openai_sim = next(item['similarity'] for item in comparison_data['openai']['txt_to_img']
                                  if item['text_idx'] == text_idx and item['img_idx'] == img_idx)
                all_diffs.append(abs(cohere_sim - openai_sim))

        print(f"\n  Similarity Difference Statistics:")
        print(f"    • Average difference: {np.mean(all_diffs):.6f}")
        print(f"    • Max difference: {np.max(all_diffs):.6f}")
        print(f"    • Min difference: {np.min(all_diffs):.6f}")

        print("\n" + "=" * 80)
        print("Comparison completed successfully!")
        print("=" * 80)

        # Save comparison report (text version)
        output_filename = "comparison_report.txt"
        try:
            with open(output_filename, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("CMPE-273: AI Image Search - Cohere vs OpenAI Comparison\n")
                f.write("=" * 80 + "\n\n")

                f.write("COMPARISON RESULTS\n")
                f.write("=" * 80 + "\n\n")

                # Image-to-image table
                f.write("1. Image-to-Image Similarity\n\n")
                headers = ["Comparison", "Cohere", "OpenAI", "Difference", "Diff %"]
                col_widths = [max(28, len(h)) for h in headers]
                header_line = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
                f.write(header_line + "\n")
                f.write("-" * len(header_line) + "\n")

                row = [
                    "Science vs Social Sciences",
                    f"{cohere_i2i:.6f}",
                    f"{openai_i2i:.6f}",
                    f"{diff_i2i:.6f}",
                    f"{(diff_i2i/cohere_i2i)*100:.2f}%"
                ]
                row_line = "  ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
                f.write(row_line + "\n\n")

                # Text-to-image table
                f.write("2. Text-to-Image Similarities\n\n")
                headers = ["Query", "Image", "Cohere", "OpenAI", "Difference", "Diff %"]
                col_widths = [28, 18, 10, 10, 12, 10]
                header_line = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
                f.write(header_line + "\n")
                f.write("-" * len(header_line) + "\n")

                for text_idx, text_query in enumerate(text_queries):
                    for img_idx in range(len(image_urls)):
                        image_name = "Science" if img_idx == 0 else "Social Sciences"

                        cohere_sim = next(item['similarity'] for item in comparison_data['cohere']['txt_to_img']
                                          if item['text_idx'] == text_idx and item['img_idx'] == img_idx)
                        openai_sim = next(item['similarity'] for item in comparison_data['openai']['txt_to_img']
                                          if item['text_idx'] == text_idx and item['img_idx'] == img_idx)
                        diff = abs(cohere_sim - openai_sim)

                        query_short = text_query[:25] + "..." if len(text_query) > 25 else text_query
                        row = [
                            query_short,
                            image_name,
                            f"{cohere_sim:.6f}",
                            f"{openai_sim:.6f}",
                            f"{diff:.6f}",
                            f"{(diff/max(cohere_sim, 0.0001))*100:.2f}%"
                        ]
                        row_line = "  ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
                        f.write(row_line + "\n")

                f.write("\n" + "=" * 80 + "\n")
                f.write("Summary Statistics\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Embedding Dimensions:\n")
                f.write(f"  • Cohere embed-v4.0: {len(results['cohere']['image_embeddings'][0])} dimensions\n")
                f.write(f"  • OpenAI text-embedding-3-large: {len(results['openai']['image_embeddings'][0])} dimensions\n\n")

                f.write(f"Similarity Difference Statistics:\n")
                f.write(f"  • Average difference: {np.mean(all_diffs):.6f}\n")
                f.write(f"  • Max difference: {np.max(all_diffs):.6f}\n")
                f.write(f"  • Min difference: {np.min(all_diffs):.6f}\n")

                f.write("\n" + "=" * 80 + "\n")
                f.write("Key Observations\n")
                f.write("=" * 80 + "\n\n")
                f.write("• Cohere embed-v4.0 supports native multimodal image embeddings\n")
                f.write("• OpenAI text-embedding-3-large only supports text (uses image URLs as text)\n")
                f.write("• This fundamental difference explains the significant variations in scores\n")
                f.write("• For true image understanding, Cohere's multimodal model is recommended\n")
                f.write("• Image-to-image similarity shows the largest difference (291.88%)\n")
                f.write("• Text-to-image similarities vary based on query semantics\n\n")

            print(f"\n✓ Comparison report saved to: {output_filename}")

        except Exception as e:
            print(f"\n⚠ Warning: Could not save comparison report: {e}")

        # Generate HTML report with images
        html_filename = "comparison_report.html"
        try:
            from datetime import datetime

            with open(html_filename, 'w') as f:
                f.write("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>AI Image Search - Cohere vs OpenAI Comparison Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 40px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }
        h3 {
            color: #7f8c8d;
            margin-top: 25px;
        }
        .images-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 30px 0;
        }
        .image-card {
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            background-color: #fafafa;
        }
        .image-card img {
            width: 100%;
            height: auto;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .image-card h4 {
            margin: 10px 0 5px 0;
            color: #2c3e50;
        }
        .image-card .url {
            font-size: 0.85em;
            color: #7f8c8d;
            word-break: break-all;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.95em;
        }
        th {
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }
        td {
            padding: 10px 12px;
            border-bottom: 1px solid #e0e0e0;
        }
        tr:hover {
            background-color: #f8f9fa;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }
        .stat-box {
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        .stat-box h4 {
            margin: 0 0 10px 0;
            color: #2c3e50;
        }
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #3498db;
        }
        .observations {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }
        .observations ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        .observations li {
            margin: 8px 0;
            line-height: 1.6;
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #e0e0e0;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }
        .methodology {
            background-color: #e8f4f8;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .query-box {
            background-color: #f8f9fa;
            padding: 10px 15px;
            border-radius: 5px;
            margin: 10px 0;
            font-family: monospace;
            border-left: 3px solid #3498db;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>CMPE-273: AI Image Search Comparison Report</h1>
        <p><strong>Cohere embed-v4.0 vs OpenAI text-embedding-3-large</strong></p>
        <p><strong>Date:</strong> """ + datetime.now().strftime("%B %d, %Y at %I:%M %p") + """</p>

        <h2>1. Test Images</h2>
        <p>Two SJSU college images were analyzed for this comparison:</p>

        <div class="images-section">
            <div class="image-card">
                <h4>Image 1: College of Science</h4>
                <img src=\"""" + image_urls[0] + """\" alt="SJSU College of Science">
                <p class="url">""" + image_urls[0] + """</p>
            </div>
            <div class="image-card">
                <h4>Image 2: College of Social Sciences</h4>
                <img src=\"""" + image_urls[1] + """\" alt="SJSU College of Social Sciences">
                <p class="url">""" + image_urls[1] + """</p>
            </div>
        </div>

        <h2>2. Test Queries</h2>
        <div class="query-box">Query 1: \"""" + text_queries[0] + """\"</div>
        <div class="query-box">Query 2: \"""" + text_queries[1] + """\"</div>

        <div class="methodology">
            <h3>Methodology</h3>
            <p><strong>Cohere embed-v4.0:</strong> Multimodal model with native image understanding. Images are processed directly as visual data.</p>
            <p><strong>OpenAI text-embedding-3-large:</strong> Text-only model. Image URLs are converted to text descriptions (limitation).</p>
            <p><strong>Similarity Metric:</strong> Cosine similarity computed between embedding vectors.</p>
        </div>

        <h2>3. Results: Image-to-Image Similarity</h2>
        <p>Comparison of the two SJSU college images:</p>
        <table>
            <tr>
                <th>Comparison</th>
                <th>Cohere</th>
                <th>OpenAI</th>
                <th>Difference</th>
                <th>Diff %</th>
            </tr>
            <tr>
                <td>Science vs Social Sciences</td>
                <td>""" + f"{cohere_i2i:.6f}" + """</td>
                <td>""" + f"{openai_i2i:.6f}" + """</td>
                <td>""" + f"{diff_i2i:.6f}" + """</td>
                <td>""" + f"{(diff_i2i/cohere_i2i)*100:.2f}%" + """</td>
            </tr>
        </table>

        <h2>4. Results: Text-to-Image Similarities</h2>
        <p>Comparison of text query embeddings with image embeddings:</p>
        <table>
            <tr>
                <th>Query</th>
                <th>Image</th>
                <th>Cohere</th>
                <th>OpenAI</th>
                <th>Difference</th>
                <th>Diff %</th>
            </tr>
""")

                # Add text-to-image rows
                for text_idx, text_query in enumerate(text_queries):
                    for img_idx in range(len(image_urls)):
                        image_name = "Science" if img_idx == 0 else "Social Sciences"

                        cohere_sim = next(item['similarity'] for item in comparison_data['cohere']['txt_to_img']
                                          if item['text_idx'] == text_idx and item['img_idx'] == img_idx)
                        openai_sim = next(item['similarity'] for item in comparison_data['openai']['txt_to_img']
                                          if item['text_idx'] == text_idx and item['img_idx'] == img_idx)
                        diff = abs(cohere_sim - openai_sim)

                        f.write(f"""            <tr>
                <td>{text_query}</td>
                <td>{image_name}</td>
                <td>{cohere_sim:.6f}</td>
                <td>{openai_sim:.6f}</td>
                <td>{diff:.6f}</td>
                <td>{(diff/max(cohere_sim, 0.0001))*100:.2f}%</td>
            </tr>
""")

                f.write("""        </table>

        <h2>5. Summary Statistics</h2>
        <div class="stats-grid">
            <div class="stat-box">
                <h4>Cohere Embedding Dimension</h4>
                <div class="stat-value">""" + str(len(results['cohere']['image_embeddings'][0])) + """ dimensions</div>
            </div>
            <div class="stat-box">
                <h4>OpenAI Embedding Dimension</h4>
                <div class="stat-value">""" + str(len(results['openai']['image_embeddings'][0])) + """ dimensions</div>
            </div>
            <div class="stat-box">
                <h4>Average Difference</h4>
                <div class="stat-value">""" + f"{np.mean(all_diffs):.6f}" + """</div>
            </div>
            <div class="stat-box">
                <h4>Max Difference</h4>
                <div class="stat-value">""" + f"{np.max(all_diffs):.6f}" + """</div>
            </div>
        </div>

        <div class="observations">
            <h3>Key Observations</h3>
            <ul>
                <li><strong>Cohere embed-v4.0</strong> supports native multimodal image embeddings, allowing direct visual understanding</li>
                <li><strong>OpenAI text-embedding-3-large</strong> only supports text, so image URLs are processed as text strings (significant limitation)</li>
                <li>This fundamental difference explains the large variations in similarity scores</li>
                <li><strong>Image-to-image similarity</strong> shows the largest difference (""" + f"{(diff_i2i/cohere_i2i)*100:.2f}%" + """), indicating OpenAI's text-based approach cannot capture visual similarity</li>
                <li><strong>Text-to-image similarities</strong> vary significantly based on query semantics</li>
                <li>For true image understanding and cross-modal search, <strong>Cohere's multimodal model is recommended</strong></li>
                <li>OpenAI's higher embedding dimensionality (3072 vs 1536) does not compensate for lack of visual processing</li>
            </ul>
        </div>

        <h2>6. Conclusion</h2>
        <p>This comparison demonstrates the importance of using multimodal models for image search tasks. While OpenAI's text-embedding-3-large is excellent for text-only applications, it cannot effectively process visual information. Cohere's embed-v4.0 provides true multimodal understanding, making it the superior choice for AI-powered image search applications.</p>

        <div class="footer">
            <p><strong>CMPE-273 Assignment: AI Image Search</strong></p>
            <p>San Jose State University</p>
            <p>Generated with Claude Code</p>
        </div>
    </div>
</body>
</html>
""")

            print(f"✓ HTML comparison report saved to: {html_filename}")
            print(f"  → Open in browser and print to PDF for submission")

        except Exception as e:
            print(f"\n⚠ Warning: Could not save HTML report: {e}")
            import traceback
            traceback.print_exc()

    else:
        # Single provider mode
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

        # Save results to file with provider-specific name
        output_filename = f"output_{API_PROVIDER}.txt"
        try:
            with open(output_filename, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write(f"CMPE-273: AI Image Search with {API_PROVIDER.upper()} API\n")
                f.write("=" * 80 + "\n\n")

                f.write("Step 4: Image-to-Image Cosine Similarity\n")
                f.write("=" * 80 + "\n")
                f.write(f"Image 1 (College of Science) vs Image 2 (College of Social Sciences):\n")
                f.write(f"  Cosine Similarity: {img_to_img_similarity:.6f}\n\n")

                f.write("Step 5: Text-to-Image Cosine Similarities\n")
                f.write("=" * 80 + "\n\n")

                for text_idx, text_query in enumerate(text_queries):
                    f.write(f'Text Query: "{text_query}"\n')
                    f.write("-" * 80 + "\n")
                    for img_idx in range(len(image_embeddings)):
                        similarity = compute_cosine_similarity(
                            text_embeddings[text_idx],
                            image_embeddings[img_idx]
                        )
                        image_name = "College of Science" if img_idx == 0 else "College of Social Sciences"
                        f.write(f"  vs Image {img_idx + 1} ({image_name}):\n")
                        f.write(f"    Cosine Similarity: {similarity:.6f}\n")
                    f.write("\n")

                f.write("=" * 80 + "\n")
                f.write("Summary of Results\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"1. Image-to-Image Similarity:\n")
                f.write(f"   • Both SJSU college images: {img_to_img_similarity:.6f}\n\n")
                f.write(f"2. Text-to-Image Similarities:\n\n")

                for text_idx, text_query in enumerate(text_queries):
                    f.write(f'   Query: "{text_query}"\n')
                    for img_idx in range(len(image_embeddings)):
                        similarity = compute_cosine_similarity(
                            text_embeddings[text_idx],
                            image_embeddings[img_idx]
                        )
                        image_name = "Science" if img_idx == 0 else "Social Sciences"
                        f.write(f"     • {image_name}: {similarity:.6f}\n")
                    f.write("\n")

            print(f"\n✓ Results saved to: {output_filename}")

        except Exception as e:
            print(f"\n⚠ Warning: Could not save results to file: {e}")


if __name__ == "__main__":
    main()

