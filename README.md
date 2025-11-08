# CMPE-273 Assignment: AI Image Search with Cohere Embeddings

## Assignment Overview

This project demonstrates AI-powered image search using **Cohere's embed-v4.0 model**. The implementation generates embeddings for images and text queries, then computes cosine similarity scores to measure semantic relationships between visual and textual content.

## Features

- **Image Embedding Generation**: Downloads images from URLs and generates embeddings using Cohere embed-v4.0
- **Text Embedding Generation**: Creates embeddings for natural language queries
- **Cosine Similarity Computation**: Measures similarity between:
  - Image-to-Image comparisons
  - Text-to-Image cross-modal searches
- **Clean Output**: Professional formatting with detailed results display

## Requirements

- Python 3.8+
- Cohere API Key

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd ai_image_search
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Cohere API key:
   - Sign up at [Cohere](https://cohere.com/)
   - Get your API key from the dashboard
   - Replace `YOUR_API_KEY_HERE` in the script with your actual API key

## Usage

Run the script:
```bash
python cohereembeddingreleasewithsearch.py
```

## Implementation Details

### Images Analyzed
1. SJSU College of Science - `https://www.sjsu.edu/_images/people/ADV_college-of-science_2.jpg`
2. SJSU College of Social Sciences - `https://www.sjsu.edu/_images/people/ADV_college-of-social-sciences_2.jpg`

### Text Queries
1. "person with tape and cap"
2. "cart with single tire"

### Key Functions

- `download_and_encode_image(url)`: Downloads an image and encodes it to base64
- `compute_cosine_similarity(embedding1, embedding2)`: Computes cosine similarity using NumPy
- `main()`: Orchestrates the entire workflow

## Output

The script provides:
- Step-by-step progress indicators
- Image-to-image similarity scores
- Text-to-image similarity scores for each query-image pair
- Comprehensive summary of all results

See [example_output.txt](example_output.txt) for sample output.

## Technical Stack

- **Cohere Python SDK**: For embedding generation
- **NumPy**: For cosine similarity computation
- **Requests**: For downloading images
- **Base64**: For image encoding

## Assignment Requirements Met

✅ Uses Cohere Python SDK with embed-v4.0 model
✅ Generates embeddings for both specified images
✅ Generates embeddings for both text queries
✅ Computes cosine similarity between images
✅ Computes cosine similarity between each text query and each image
✅ Displays results in a clean, professional format
✅ Code is runnable with API key replacement

## Author

CMPE-273 Student
San Jose State University
Date: November 8, 2025

## License

This project is submitted as part of CMPE-273 coursework at SJSU.
