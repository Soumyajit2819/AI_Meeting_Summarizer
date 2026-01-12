**AI Meeting Summarizer**:

The AI Meeting Summarizer converts audio recordings from meetings (local files) into structured, actionable summaries.
It is designed to help teams quickly identify key points, decisions, and action items from long meetings.



**Key Features**:

**Audio Transcription**: Uses Whisper to convert audio (MP3, WAV, M4A, MP4) to text.

**Entity Extraction**: Uses SpaCy to extract important entities like people, dates, and organizations.

**Sentence Classification**:

TF-IDF + SGDClassifier trained on meeting-specific labeled data

**Categorizes sentences into**:

Tasks & Action Items

Key Decisions & Plans

Issues, Risks & Costs

Logistics & Schedule

**Rule-Based Patterns**: Detects tasks, decisions, issues, and logistics using regex patterns for higher accuracy.

**Formatted Summaries**: Outputs clean, structured text including participants, dates, and categorized content.



**Training the Model**:


Dataset: data/train_dataset.csv (labeled meeting sentences)

Script: train_data.py

Output: whisper_meeting_classifier.pkl


**Testing the Model**:


Audio file: meeting_audio.mp3

Script: test.py

Output: result.txt (summary saved in data/)


