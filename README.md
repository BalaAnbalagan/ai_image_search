# CMPE-273 Assignment: AI Image Search with Multi-Provider Support

## Assignment Overview

This project demonstrates AI-powered image search with support for multiple embedding providers. The implementation generates embeddings for images and text queries, then computes cosine similarity scores to measure semantic relationships between visual and textual content.

**Supported Providers:**
- **Cohere embed-v4.0** (Recommended - matches assignment requirements)
- **OpenAI text-embedding-3-large** (Alternative option)

## Features

- **Multi-Provider Support**: Choose between Cohere or OpenAI APIs
- **Image Embedding Generation**: Downloads images from URLs and generates embeddings
- **Text Embedding Generation**: Creates embeddings for natural language queries
- **Cosine Similarity Computation**: Measures similarity between:
  - Image-to-Image comparisons
  - Text-to-Image cross-modal searches
- **Clean Output**: Professional formatting with detailed results display
- **Easy Configuration**: Simple one-line switch between providers

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

3. Configure your API provider:
   - Open `ai_image_search.py`
   - Set `API_PROVIDER = "cohere"` or `"openai"` (line 138)
   - Add your API key to the corresponding variable (lines 141-142)

See **[API_CONFIGURATION_GUIDE.md](API_CONFIGURATION_GUIDE.md)** for detailed setup instructions.

## Usage

### Quick Start (Cohere - Recommended)
```python
# In ai_image_search.py
API_PROVIDER = "cohere"
COHERE_API_KEY = "your-cohere-key-here"
```

Then run:
```bash
python ai_image_search.py
```

### Alternative (OpenAI)
```python
# In ai_image_search.py
API_PROVIDER = "openai"
OPENAI_API_KEY = "your-openai-key-here"
```

**Note**: OpenAI doesn't support direct image embeddings, so results will be less accurate for this use case.

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
