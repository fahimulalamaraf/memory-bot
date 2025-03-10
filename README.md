# MemoryBot

MemoryBot is a web application designed to manage and search photos using advanced AI capabilities. It leverages FastAPI for the backend, Qdrant for vector storage, and various AI services for image and text processing.

## Features

- **Image Upload**: Upload images with metadata such as who, where, when, and description.
- **Image Search**: Search images using natural language queries, metadata, or by providing an image.
- **Contextual Search**: Combines image content and user queries to perform a refined search.
- **Chat Interface**: Interact with MemoryBot through a chat interface for personalized assistance.
- **Rate Limiting**: Protects the API from abuse by limiting the number of requests.

## Technologies Used

- **FastAPI**: A modern, fast (high-performance) web framework for building APIs with Python 3.7+.
- **Qdrant**: A vector similarity search engine for storing and querying image embeddings.
- **CLIP**: A model from OpenAI for generating image and text embeddings.
- **Gemini**: A service for generating content and analyzing images.
- **PIL**: Python Imaging Library for image processing.
- **Jinja2**: A templating engine for rendering HTML templates.

## Installation

### Prerequisites

- Python 3.8+
- Virtualenv (optional but recommended)
- Qdrant server

### Setup

1. **Clone the repository**:
   ```bash
   git clone
   cd memorybot
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   - Create a `.env` file in the root directory.
   - Add the following variables:
     ```
     GEMINI_API_KEY=your_gemini_api_key
     ```

5. **Run Qdrant**:
   - Ensure Qdrant is running locally or accessible from your application.

6. **Start the application**:
   ```bash
   python main.py
   ```
## Project Structure
```
IMAGE_GALLERY_APP/
│
├── memorybot/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── __init__.py
│   │   └── endpoints/
│   │       ├── __init__.py
│   │       ├── chat.py
│   │       └── upload.py
│   │       └── gallery.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── app_factory.py
│   │   ├── cache.py
│   │   ├── config.py
│   │   ├── constants.py
│   │   ├── database.py
│   │   ├── gemini_cache.py
│   │   ├── logging_config.py
│   │   └── middleware.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── image.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── pages.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── chat_service.py
│   │   ├── clip_service.py
│   │   ├── gemini_service.py
│   │   ├── image_processor.py
│   │   └── qdrant_service.py
│   │
│   ├── static/
│   │   └── css/
│   │       └── shared.css
│   │
│   └── templates/
│       ├── base.html
│       ├── image-detail.html
│       ├── index.html
│       ├── premium-homepage.html
│       ├── premium-gallery.html
│       ├── upload.html
│       └── chat-interface.html
│
├── images/
│
├── qdrant_storage/
│
├── venv/
│
├── .env
├── main.py
├── README.md
├── requirements.txt
└── research&Development.py
```

## Usage

### Access the Application

- Run the application by running
  ```
    python main.py
  ```

### API Endpoints

- **Upload Image**: `POST /upload`
- **Search Images**: `GET /api/v1/gallery/search`
- **Chat with MemoryBot**: `POST /chat`
- **Get Image Details**: `GET /image/{filename}`

### Testing

- Use tools like Postman or curl to test the API endpoints.
- Ensure the application is running before testing.

## Project Structure

- `main.py`: Entry point of the application.
- `app/api`: Contains API routers and endpoints.
- `app/core`: Core configurations, middleware, and utilities.
- `app/models`: Data models for the application.
- `app/services`: Services for handling business logic and integrations.
- `app/static`: Static files like CSS and JavaScript.
- `app/templates`: HTML templates for rendering views.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for the CLIP model.
- Qdrant for the vector search engine.
- FastAPI for the web framework.
- Contributors and community for their support and contributions.
