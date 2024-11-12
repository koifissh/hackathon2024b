
![PNG image](https://github.com/user-attachments/assets/3f7dfc20-8b13-49f2-a335-192db3bb7bf9)
# Emergency 911 Dispatch System
A real-time emergency dispatch interface with speech recognition, AI assistance, and location tracking.

## Authors
Daniel Huynh and Yagna Patel

## Problem
Traditional emergency dispatch systems often lack real-time transcription, automated assistance, and visual tracking capabilities. This system addresses these limitations by providing an integrated solution for emergency call handling and dispatch management.

## Statement
This system can be used by emergency dispatch centers to enhance their response capabilities through AI-assisted call handling, automated location detection, and real-time status tracking. The interface helps dispatchers manage emergency calls more efficiently while maintaining accurate records and providing visual feedback.

## Usage
To use this program, follow these steps:

1. Install the required dependencies:
```python
pip install flask
pip install flask-socketio
pip install sounddevice
pip install openai
pip install numpy
pip install wave
```

2. Ensure you have valid OpenAI API credentials configured in the EmergencyDispatcher class.

3. Run the `Gui-1.py` file to start the server.

4. Access the interface through a web browser at `localhost:5000`.

5. The interface provides the following features:
   - Real-time speech recognition and transcription
   - AI-assisted response generation
   - Automatic location detection and mapping
   - Emergency type classification
   - Unit dispatch tracking
   - Call status monitoring

## Additional Details
- The system uses Flask and Socket.IO for real-time web communication
- OpenAI's Whisper model handles speech-to-text conversion
- OpenAI's GPT model provides AI-assisted responses via GPT "Assistants"
- OpenStreetMap integration for location visualization
- Real-time updates for transcript, dispatch status, and emergency summaries

- Key libraries and services used:
   - `Flask: Web framework`
   - `Socket.IO: Real-time communication`
   - `OpenAI API: Speech recognition and AI assistance`
   - `Sounddevice: Audio processing`
   - `Leaflet.js: Map visualization`
   - `OpenStreetMap: Geocoding services`

## Limits
- Requires stable internet connection for API services
- Speech recognition accuracy depends on audio quality
- Location detection relies on address mention in conversation
- Limited to predefined emergency types (Medical, Fire, Police)
- Requires OpenAI API key and credits

## Strengths
- Real-time processing and updates
- Automated speech recognition and transcription
- Intelligent response generation
- Visual mapping and location tracking
- Comprehensive emergency type detection
- Automated unit dispatch suggestions

## Expansions
- Add support for multiple concurrent calls
- Implement direct emergency service integration
- Add video call capabilities
- Expand emergency type classifications
- Integrate with external dispatch systems
- Add historical call data analysis

## Complexity
**Time Complexity for Key Operations**
- Speech detection: O(n) where n is the audio chunk size
- Address detection: O(n) where n is the transcript length
- Emergency type detection: O(1) using pattern matching
- Location geocoding: O(1) API call
- Real-time updates: O(1) per event

**Space Complexity for Key Components**
- Audio buffer: O(n) where n is the recording duration
- Transcript history: O(n) where n is the conversation length
- Emergency summary: O(1) fixed size structure
- Map data: O(1) single location tracking
- Dispatch status: O(m) where m is the number of dispatched units

## Citations
https://csgjusticecenter.org/publications/911-dispatch-call-processing-protocols-key-tools-for-coordinating-effective-call-triage/
