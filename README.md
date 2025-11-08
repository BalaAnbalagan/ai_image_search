# CMPE-273 Assignment: AI Image Search with Multi-Provider Support

## Assignment Overview

This project demonstrates AI-powered image search with support for multiple embedding providers. The implementation generates embeddings for images and text queries, then computes cosine similarity scores to measure semantic relationships between visual and textual content.

**Supported Providers:**
- **Cohere embed-v4.0** (Multimodal - supports both images and text)
- **OpenAI text-embedding-3-large** (Text-only - not used for image embeddings)

## Features

- **Multi-Provider Support**: Choose between Cohere or OpenAI APIs
- **Comparison Mode**: Run both providers simultaneously and compare results side-by-side
- **Automatic Image Detection**: Uses 2 images (assignment mode) or 10+ images (extended mode)
- **Image Embedding Generation**: Downloads images from URLs and generates embeddings
- **Text Embedding Generation**: Creates embeddings for natural language queries
- **Cosine Similarity Computation**: Measures similarity between:
  - Image-to-Image comparisons
  - Text-to-Image cross-modal searches
- **Interactive Search**: Search with any text query and get ranked results
- **Clean Output**: Professional formatting with detailed results display
- **Secure API Key Management**: Uses .env file to keep credentials safe

## Requirements

- Python 3.8+
- API Key (Cohere recommended, or OpenAI)

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

3. Configure your API keys:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your API key(s):
     ```
     COHERE_API_KEY=your-cohere-key-here
     OPENAI_API_KEY=your-openai-key-here
     ```

See **[API_CONFIGURATION_GUIDE.md](API_CONFIGURATION_GUIDE.md)** for detailed setup instructions.

## Usage

### Jupyter Notebook (Recommended)

Interactive notebook covering both learning and practical application:

```bash
jupyter notebook ai_image_search.ipynb
```

**Part 1: Core Concepts** (Satisfies requirements)
- Uses the 2 specific SJSU images
- Demonstrates `embed-v4.0` model
- Computes image-to-image similarity
- Uses the 2 specified text queries
- Shows text-to-image similarities
- Explains embedding concepts

**Part 2: Practical Search Engine** (Real-world extension)
- Load images from **any folder**
- Search with **text input queries**
- Get **ranked results** with scores
- Try **unlimited searches** interactively

See **[NOTEBOOK_GUIDE.md](NOTEBOOK_GUIDE.md)** for detailed usage instructions.

### Single Provider Mode (Command Line)

Run the script:
```bash
python3 ai_image_search.py
```

If you have **one API key** configured:
- The script will automatically use that provider

If you have **both API keys** configured, you'll see a menu:
```
1. COHERE only - Cohere embed-v4.0 (Recommended for assignment)
2. OPENAI only - OpenAI text-embedding-3-large
3. COMPARE BOTH - Run both and generate comparison report
```

### Comparison Mode (Command Line)

Select option **3** to run both providers and generate a detailed comparison report. See **[COMPARISON_MODE_GUIDE.md](COMPARISON_MODE_GUIDE.md)** for details.

**Note**: OpenAI doesn't support direct image embeddings, so it uses image URLs as text (limitation of the API).

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
