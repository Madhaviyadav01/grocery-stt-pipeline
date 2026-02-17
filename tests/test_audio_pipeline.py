
import unittest
import numpy as np
import os
import shutil
import soundfile as sf
from pipeline.Audio_preprocessing.preprocess import PreprocessPipeline

class TestAudioPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_data"
        self.output_dir = "test_output"
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a dummy sine wave audio file
        sr = 16000
        t = np.linspace(0, 1, sr) # 1 second
        audio = 0.5 * np.sin(2 * np.pi * 440 * t) # 440 Hz sine wave
        
        # Add silence
        silence = np.zeros(sr // 2) # 0.5 seconds silence
        audio_with_silence = np.concatenate([silence, audio, silence])
        
        self.test_file = os.path.join(self.test_dir, "test_sine.wav")
        sf.write(self.test_file, audio_with_silence, sr)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_pipeline_flow(self):
        pipeline = PreprocessPipeline(target_sr=16000)
        output_path = os.path.join(self.output_dir, "processed_sine.wav")
        
        # Process the file
        processed_audio, sr = pipeline.process_file(self.test_file, output_path)
        
        # Checks
        self.assertEqual(sr, 16000)
        self.assertTrue(len(processed_audio) < 2 * 16000) # Should be shorter due to trimming
        self.assertTrue(len(processed_audio) > 0)
        self.assertTrue(os.path.exists(output_path))
        
        print("Test passed: Pipeline processed audio successfully.")

if __name__ == "__main__":
    unittest.main()
