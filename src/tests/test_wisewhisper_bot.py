import unittest
from src.wisewhisper_bot import process_with_llm

class TestWiseWhisperBot(unittest.TestCase):
    def test_process_with_llm(self):
        response = process_with_llm("Tell me a fun fact about cats.")
        self.assertIn("cat", response.lower())  # Checks if "cat" is mentioned in the response

if __name__ == "__main__":
    unittest.main()
