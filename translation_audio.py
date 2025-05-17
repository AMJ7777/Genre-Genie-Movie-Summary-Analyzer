"""
    Translation and audio module for the Movie Summary Analysis application.
Handles text translation and text-to-speech conversion.
"""
import os
import time
import pygame
from deep_translator import GoogleTranslator
from gtts import gTTS
import nltk
nltk.download('punkt')
# Directory paths
DATA_DIR = "data"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)
class TranslationAudioManager:
    """Class for managing translation and text-to-speech functionality."""
    
    def __init__(self, callback=None):
        """Initialize the TranslationAudioManager.
        
        Args:
            callback: A function to call with progress updates.
        """
        self.callback = callback
        pygame.mixer.init()
        self.current_audio = None
        
        # Language codes
        self.languages = {
            'English': 'en',
            'Arabic': 'ar',
            'Urdu': 'ur',
            'Korean': 'ko'
        }
    
    def translate_text(self, text, target_language):
        """Translate text to the target language.
        
        Args:
            text: The text to translate.
            target_language: The target language code.
            
        Returns:
            Translated text.
        """
        if target_language == 'en':
            return text
        
        if self.callback:
            self.callback(0, "Translating text...")
        
        try:
            # Translate the text
            translator = GoogleTranslator(source='auto', target=target_language)
            translated_text = translator.translate(text)
            
            if self.callback:
                self.callback(100, "Translation complete")
            
            return translated_text
        except Exception as e:
            print(f"Translation error: {e}")
            
            if self.callback:
                self.callback(100, f"Translation error: {e}")
            
            return None
    
    def text_to_speech(self, text, language_code, filename=None):
        """Convert text to speech.
        
        Args:
            text: The text to convert.
            language_code: The language code.
            filename: The filename to save the audio to.
            
        Returns:
            Path to the saved audio file.
        """
        if self.callback:
            self.callback(0, "Converting to speech...")
        
        try:
            # Create a unique filename if none provided
            if filename is None:
                timestamp = int(time.time())
                filename = f"tts_{language_code}_{timestamp}.mp3"
            
            # Create file path
            file_path = os.path.join(AUDIO_DIR, filename)
            
            # Convert text to speech
            tts = gTTS(text=text, lang=language_code, slow=False)
            
            if self.callback:
                self.callback(50, "Saving audio file...")
            
            # Save to file
            tts.save(file_path)
            
            if self.callback:
                self.callback(100, "Audio conversion complete")
            
            return file_path
        except Exception as e:
            print(f"Text-to-speech error: {e}")
            
            if self.callback:
                self.callback(100, f"Text-to-speech error: {e}")
            
            return None
    
    def play_audio(self, file_path):
        """Play an audio file.
        
        Args:
            file_path: Path to the audio file.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Stop any currently playing audio
            self.stop_audio()
            
            # Load and play the new audio
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            self.current_audio = file_path
            
            return True
        except Exception as e:
            print(f"Audio playback error: {e}")
            return False
    
    def stop_audio(self):
        """Stop any currently playing audio."""
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            self.current_audio = None
    
    def is_playing(self):
        """Check if audio is currently playing.
        
        Returns:
            True if audio is playing, False otherwise.
        """
        return pygame.mixer.music.get_busy()
    
    def translate_and_speak(self, text, language):
        """Translate text and convert to speech.
        
        Args:
            text: The text to translate and speak.
            language: The target language.
            
        Returns:
            Path to the saved audio file.
        """
        # Get language code
        language_code = self.languages.get(language, 'en')
        
        # Translate text
        translated_text = self.translate_text(text, language_code)
        
        if translated_text:
            # Convert to speech
            audio_path = self.text_to_speech(translated_text, language_code)
            
            return audio_path
        
        return None
    
    def save_translated_summary(self, summary, movie_id):
        """Save a summary translated to all available languages.
        
        Args:
            summary: The summary to translate.
            movie_id: The ID of the movie.
            
        Returns:
            Dictionary mapping language names to audio file paths.
        """
        audio_files = {}
        
        # Translate and convert to speech for each language
        for language, code in self.languages.items():
            if self.callback:
                self.callback(0, f"Processing {language}...")
            
            # Translate
            translated_text = self.translate_text(summary, code)
            
            if translated_text:
                # Generate filename
                filename = f"movie_{movie_id}_{code}.mp3"
                
                # Convert to speech
                audio_path = self.text_to_speech(translated_text, code, filename)
                
                if audio_path:
                    audio_files[language] = audio_path
            
            if self.callback:
                self.callback(100, f"{language} processing complete")
        
        return audio_files