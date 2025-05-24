import re  
import joblib 
import spotipy  
from spotipy.oauth2 import SpotifyClientCredentials  
from flask import Flask, request, jsonify, send_from_directory  
from flask_cors import CORS  
import time  
import logging  
import json  
import os 

app = Flask(__name__)  
CORS(app)  


logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__) 

# 2. Load configuration from config.json
CONFIG_PATH = "config.json"  # Path to configuration file
# Check if config file exists
if not os.path.exists(CONFIG_PATH):
    logger.error(f"Configuration file {CONFIG_PATH} not found")  # Log error if file is missing
    raise FileNotFoundError(f"Configuration file {CONFIG_PATH} not found")  # Raise exception
try:
    # Read and parse JSON configuration file
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    # Extract search terms and markets from config
    search_terms = config.get("search_terms") 
    markets = config.get("markets") 
    # Validate required fields
    if not search_terms or not markets:
        logger.error("Missing required fields in config.json: 'search_terms' and 'markets' are required")
        raise ValueError("Missing required fields in config.json: 'search_terms' and 'markets' are required")
except json.JSONDecodeError as e:
    logger.error(f"Invalid JSON in configuration file: {str(e)}")  # Log JSON parsing errors
    raise ValueError(f"Invalid JSON in configuration file: {str(e)}")
except Exception as e:
    logger.error(f"Error loading configuration: {str(e)}")  # Log other configuration errors
    raise ValueError(f"Error loading configuration: {str(e)}")

# 3. Load pre-trained emotion detection model and vectorizer
model = joblib.load('emotion_model.pkl')
# Load  TF-IDF vectorizer for text feature extraction
vectorizer = joblib.load('vectorizer.pkl')

def preprocess(text):

    text = text.lower()  # Convert to lowercase for consistency
    text = re.sub(r"http\S+", "", text)  # Remove URLs (e.g., http://example.com)
    text = re.sub(r"@\w+", "", text)  # Remove mentions (e.g., @username)
    text = re.sub(r"#\w+", "", text)  # Remove hashtags (e.g., #happy)
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation and special characters
    text = re.sub(r"\d+", "", text)  # Remove digits (e.g., 123)
    return text.strip()  # Remove leading/trailing whitespace

# 5. Define emotion detection function
def detect_emotion(text):

    clean_text = preprocess(text)  # Clean the input text
    features = vectorizer.transform([clean_text])
    # Predict emotion using the trained model; extract single prediction
    prediction = model.predict(features)[0]
    return prediction

# 6. Set up Spotify API client
def create_spotify_client():

    # Get client ID and secret from environment variables, with defaults
    client_id = os.getenv("SPOTIFY_CLIENT_ID", "4bfa5a1de68b48fa94ddae8873ce0262")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "92adeb871f9441719809300380b646a4")
    
    try:
        # Initialize client credentials manager for OAuth
        auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        # Create and return Spotify client
        return spotipy.Spotify(auth_manager=auth_manager)
    except Exception as e:
        logger.error(f"Spotify connection error: {str(e)}")  # Log authentication errors
        return None

# 7. Retrieve track metadata from Spotify
def getTrackFeatures(id, market):

    sp = create_spotify_client()  # Create Spotify client
    if not sp:
        return None  # Return None if client creation fails
    
    try:
        # Fetch track details from Spotify API
        track_info = sp.track(id, market=market)
        # Extract relevant metadata
        name = track_info['name']
        album = track_info['album']['name']
        artist = track_info['artists'][0]['name']  # Use first artist
        url = track_info['external_urls']['spotify']  # Spotify track URL
        preview_url = track_info.get('preview_url')  # Preview URL (may be None)
        # Create dictionary with track data
        track_data = {
            'name': name,
            'album': album,
            'artist': artist,
            'url': url,
            'preview_url': preview_url
        }
        return track_data
    except Exception as e:
        logger.error(f"Track features error for track {id} in market {market}: {str(e)}")  # Log errors
        return None

# 8. Search Spotify for tracks matching mood
def search_tracks_by_mood(sp, emotion, music_type, limit=25):

    try:
        # Get search terms for the music type and emotion from config
        terms = search_terms.get(music_type, {}).get(emotion, [])
        # Get market code, default to US if not found
        market = markets.get(music_type, "US")
        if not terms:
            logger.warning(f"No search terms defined for {music_type} - {emotion}")  # Log missing terms
            return []  # Return empty list if no terms
        
        all_tracks = []  # Store all found tracks
        for term in terms:
            for attempt in range(2):  # Retry once on failure
                try:
                    # Search Spotify for tracks matching term, year 2000–2024, in specified market
                    results = sp.search(q=f"{term} year:2000-2024", type="track", limit=15, market=market)
                    # Extract tracks from results if available
                    if results and 'tracks' in results and 'items' in results['tracks']:
                        for track in results['tracks']['items']:
                            all_tracks.append(track)  # Add track to list
                    break  # Exit retry loop on success
                except Exception as e:
                    logger.warning(f"Search attempt {attempt+1} failed for term {term}: {str(e)}")  # Log retry failure
                    time.sleep(1)  # Wait 1 second before retrying
                    if attempt == 1:
                        logger.error(f"Failed to search for term {term}: {str(e)}")  # Log final failure
        
        # Remove duplicate tracks by ID
        unique_tracks = []
        seen_ids = set()
        for track in all_tracks:
            if track['id'] not in seen_ids:
                seen_ids.add(track['id'])
                unique_tracks.append(track)
                
        logger.info(f"Search found {len(unique_tracks)} tracks for {music_type} - {emotion} in market {market}")
        return unique_tracks
    except Exception as e:
        logger.error(f"Search tracks error: {str(e)}")  # Log general search errors
        return []

# 9. Recommend songs based on emotion and music type
def recommend_songs(emotion, music_type, limit=20):

    sp = create_spotify_client()  # Create Spotify client
    if not sp:
        return "Error connecting to Spotify", [], []  # Return error if client creation fails

    try:
        # Search for tracks matching emotion and music type
        all_tracks = search_tracks_by_mood(sp, emotion, music_type, limit=30)
        market = markets.get(music_type, "US")  # Get market code
        if not all_tracks:
            logger.error(f"No tracks found in search for {music_type} - {emotion}")  # Log no results
            return "No tracks found for your emotion", [], []  # Return error
        
        recommended_songs = []  # Store recommended track metadata
        for track in all_tracks:
            # Fetch track details
            track_data = getTrackFeatures(track['id'], market)
            if track_data:
                recommended_songs.append(track_data)  # Add valid track data

        # Create formatted song list for display (e.g., "Song Name - Artist")
        song_list = [f"{song['name']} - {song['artist']}" for song in recommended_songs]
        logger.info(f"Returning {len(recommended_songs)} songs for {music_type} - {emotion}")
        return "Songs found!", song_list, recommended_songs  # Return success with results

    except Exception as e:
        logger.error(f"Recommend songs error: {str(e)}")  # Log general errors
        return f"Error: {str(e)}", [], []  # Return error with empty results

# 10. Serve the frontend HTML file
@app.route('/')
def serve_index():

    return send_from_directory('.', 'index.html')  # Serve index.html from current directory

# 11. Define API endpoint for emotion detection and song recommendation
@app.route('/detect_emotion', methods=['POST'])
def api_detect_emotion():

    data = request.get_json()  # Parse JSON payload from request
    text = data.get('text', '')  # Extract text, default to empty string
    music_type = data.get('music_type', 'Pakistani')  # Extract music type, default to Pakistani
    
    # Validate input
    if not text:
        return jsonify({'error': 'No text provided'}), 400  # Return 400 error if text is empty
    if music_type not in ['Pakistani', 'Bollywood', 'Hollywood']:
        return jsonify({'error': 'Invalid music type'}), 400  # Return 400 error if music type is invalid
    
    # Detect emotion from text
    emotion = detect_emotion(text)
    # Ensure emotion is explicitly Happy or Sad (redundant but safe)
    emotion_text = "Happy" if emotion == "Happy" else "Sad"
    print(f"Detected Emotion: {emotion_text}, Music Type: {music_type}")  # Log to console
    logger.info(f"Detected Emotion: {emotion_text}, Music Type: {music_type}")  # Log to file
    
    # Get song recommendations
    status, song_list, songs_with_preview = recommend_songs(emotion_text, music_type)
    
    # Return JSON response with results
    return jsonify({
        'emotion': emotion_text,
        'music_type': music_type,
        'status': status,
        'songs': song_list,
        'songs_with_preview': songs_with_preview
    })

# 12. Run the Flask application
if __name__ == "__main__":
    # Start the Flask development server on port 5000 if script is run directly
    app.run(port=5000)