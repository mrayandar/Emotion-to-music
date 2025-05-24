# Emotion to Music

A web application that detects emotions from text input and recommends music based on the detected emotion and preferred music genre (Pakistani, Bollywood, or Hollywood).

## Features

- Text-based emotion detection (Happy/Sad)
- Music recommendations from three different genres:
  - Pakistani
  - Bollywood
  - Hollywood
- Spotify integration for music previews
- Real-time emotion analysis
- Modern web interface

## Prerequisites

- Python 3.x
- Spotify Developer Account (for API access)
- Required Python packages (install using `pip install -r requirements.txt`):
  - Flask
  - Flask-CORS
  - spotipy
  - scikit-learn
  - joblib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mrayandar/Emotion-to-music
cd Emotion_To_Music
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up Spotify API credentials:
   - Create a Spotify Developer account at https://developer.spotify.com
   - Create a new application to get Client ID and Client Secret
   - Set environment variables:
     ```bash
     export SPOTIFY_CLIENT_ID="your_client_id"
     export SPOTIFY_CLIENT_SECRET="your_client_secret"
     ```

## Project Structure

- `main.py` - Main Flask application and API endpoints
- `train.py` - Script for training the emotion detection model
- `test.py` - Testing script for the emotion detection model
- `config.json` - Configuration file for music search terms and market settings
- `emotion_model.pkl` - Pre-trained emotion detection model
- `vectorizer.pkl` - TF-IDF vectorizer for text processing
- `index.html` - Frontend web interface

## Usage

1. Start the Flask server:
```bash
python main.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Enter your text in the input field and select your preferred music genre.

4. The application will:
   - Detect the emotion from your text
   - Recommend songs based on the detected emotion and selected genre
   - Provide Spotify preview links for the recommended songs

## API Endpoints

- `GET /` - Serves the main web interface
- `POST /detect_emotion` - Detects emotion and returns music recommendations
  - Request body:
    ```json
    {
        "text": "Your text here",
        "music_type": "Pakistani|Bollywood|Hollywood"
    }
    ```

## Model Training

The emotion detection model can be trained using the provided training script:
```bash
python train.py
```

## License

[Your License]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 