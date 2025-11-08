# AI Image Search with Multimodal Embeddings

A practical implementation of AI-powered image search using multimodal embeddings. This project demonstrates semantic search capabilities by comparing images and text queries using vector embeddings and cosine similarity.

**Author:** Bala Anbalagan
**Email:** bala.anbalagan@sjsu.edu

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BalaAnbalagan/ai_image_search/blob/main/ai_image_search.ipynb)
[![View in nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.org/github/BalaAnbalagan/ai_image_search/blob/main/ai_image_search.ipynb)

## Features

- **Multimodal Embeddings**: Generate embeddings for both images and text using Cohere's embed-v4.0 model
- **Semantic Image Search**: Find images using natural language queries
- **Image-to-Image Similarity**: Compare and rank image similarities
- **Interactive Jupyter Notebook**: Learn embedding concepts with hands-on examples
- **Automatic Mode Detection**: Works with 2 images (demo mode) or folders with multiple images
- **Clean Visualizations**: Grid layouts and side-by-side comparisons
- **Secure API Management**: Environment-based configuration

## Supported Providers

- **Cohere embed-v4.0** (Multimodal - supports both images and text)
- **OpenAI text-embedding-3-large** (Text-only - not used for image embeddings)

## Quick Start

### Prerequisites

- Python 3.8+
- Cohere API key ([Get free key](https://dashboard.cohere.com/api-keys))

### Installation

1. Clone the repository:
```bash
git clone https://github.com/BalaAnbalagan/ai_image_search.git
cd ai_image_search
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API key:
   - Copy `.env.example` to `.env`
   - Add your Cohere API key to `.env`:
     ```
     COHERE_API_KEY=your-key-here
     ```

### Usage

#### Jupyter Notebook (Recommended)

The interactive notebook provides a complete learning experience. You can run it in multiple ways:

**Option 1: Google Colab (No setup required)**
- Click the "Open in Colab" badge above
- Add your Cohere API key when prompted
- Run all cells

**Option 2: Jupyter Locally**
```bash
jupyter notebook ai_image_search.ipynb
```

**Option 3: View Only**
- Click the "nbviewer" badge to see a static rendered version

**What you'll learn:**
- How embeddings represent visual and textual content
- Computing similarity between images
- Cross-modal search (text-to-image)
- Building a practical image search engine

**Features:**
- Automatic image detection (2 images or folder mode)
- Visual comparisons with side-by-side displays
- Interactive search with ranked results
- Educational explanations of embedding concepts

#### Command Line

For quick testing:

```bash
python3 ai_image_search.py
```

## Project Structure

```
ai_image_search/
├── ai_image_search.ipynb    # Main interactive notebook
├── requirements.txt          # Python dependencies
├── .env.example             # API key template
├── images/                  # Sample images
└── README.md
```

## How It Works

1. **Image Encoding**: Images are converted to base64 format
2. **Embedding Generation**: Cohere's embed-v4.0 model creates vector representations
3. **Similarity Computation**: Cosine similarity measures semantic relationships
4. **Search & Ranking**: Text queries are matched against image embeddings

## Technical Details

### Key Technologies

- **Cohere Python SDK**: Multimodal embedding generation
- **NumPy**: Vector operations and cosine similarity
- **Jupyter**: Interactive development environment
- **Pillow**: Image processing
- **Pandas**: Data analysis and visualization

### Similarity Scoring

Cosine similarity scores range from -1 to 1:
- **0.8 - 1.0**: Excellent match
- **0.6 - 0.8**: Good match
- **0.4 - 0.6**: Moderate match
- **0.2 - 0.4**: Weak match
- **< 0.2**: Poor match

## Example Use Cases

- **Content-Based Image Retrieval**: Find similar images in large collections
- **Visual Search**: Search image databases with natural language
- **Image Clustering**: Group similar images automatically
- **Duplicate Detection**: Identify near-duplicate images

## Resources

- [Cohere Embed v4.0 Documentation](https://docs.cohere.com/docs/embed-v4)
- [Image Embeddings Guide](https://docs.cohere.com/v2/docs/embeddings#image-embeddings)
- [Vector Search Tutorial](https://docs.cohere.com/docs/semantic-search)

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Sample images from [Pexels.com](https://pexels.com) (free to use under Pexels License)
- Cohere for providing the multimodal embedding API
