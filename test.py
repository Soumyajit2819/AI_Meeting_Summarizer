from whisper_transcribe import transcribe_audio
from meeting_summarizer import MeetingSummarizer

# Initialize the summarizer
summarizer = MeetingSummarizer()

# Transcribe a new local audio file
audio_file = "/Users/AI_Meeting_summarizer/backend/meeting_audio.mp3"
transcribed_text = transcribe_audio(audio_file)

transcribed_text = transcribe_audio(audio_file)

# Process the meeting
summary_results = summarizer.process_meeting(transcribed_text)

# Get formatted summary
formatted_summary = summarizer.format_summary(summary_results)

# Print to console (optional)
print(formatted_summary)

# Save to file
output_file = "result.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(formatted_summary)

print(f"\n✅ Summary saved to {output_file}")
