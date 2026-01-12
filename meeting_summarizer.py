import spacy
import re
import joblib
import os
from nlp_process import preprocess_text

class MeetingSummarizer:
    def __init__(self, model_path="whisper_meeting_classifier.pkl"):
        print("Loading NLP Models... (This happens only once)")
        
        # Load spacy for entity extraction
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load YOUR trained TF-IDF + SGDClassifier model
        # Handle both relative and absolute paths
        if not os.path.isabs(model_path):
            # If relative path, look in script directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(current_dir, model_path)
            if os.path.exists(full_path):
                model_path = full_path
        
        print(f"Loading trained model from {model_path}...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"❌ Model file not found!\n"
                f"   Looking for: {model_path}\n"
                f"   Current directory: {os.getcwd()}\n"
                f"   Please ensure 'whisper_meeting_classifier.pkl' is in the backend folder."
            )
        
        self.model = joblib.load(model_path)
        print("✅ Trained model loaded successfully!")
        
        self.label_mapping = {
            "task": "Tasks & Action Items",
            "action_item": "Tasks & Action Items",
            "decision": "Key Decisions & Plans",
            "plan": "Key Decisions & Plans",
            "issue": "Issues, Risks & Costs",
            "risk": "Issues, Risks & Costs",
            "logistics": "Logistics & Schedule",
            "schedule": "Logistics & Schedule",
            "chitchat": "Chit-chat",
            "small_talk": "Chit-chat",
        }

    def is_meeting_content(self, text: str) -> tuple:
        """
        Validate if content is actually a meeting/professional discussion
        Returns: (is_meeting, keyword_count, found_keywords)
        """
        meeting_keywords = [
            'meeting', 'agenda', 'minutes', 'discuss', 'discussion',
            'budget', 'department', 'office', 'manager', 'team',
            'schedule', 'presentation', 'proposal', 'board',
            'project', 'colleague', 'stakeholder', 'client',
            'decision', 'action item', 'follow up', 'quarterly',
            'review', 'strategy', 'plan', 'objective', 'goal',
            'deadline', 'milestone', 'deliverable', 'task',
            'attendee', 'participant', 'chair', 'facilitator',
            'conference', 'call', 'session', 'workshop', 'assigned',
            'responsible', 'approval', 'approve', 'agreed', 'consensus'
        ]
        
        text_lower = text.lower()
        
        # Count keywords in first 1000 characters (early validation)
        sample = text_lower[:1000]
        sample_keywords = [kw for kw in meeting_keywords if kw in sample]
        sample_keyword_count = len(sample_keywords)
        
        # Count keywords in full text
        total_keywords = [kw for kw in meeting_keywords if kw in text_lower]
        total_keyword_count = len(total_keywords)
        
        # Validation rules:
        # 1. Must have at least 3 keywords in first 1000 chars
        # 2. Must have at least 5 keywords total
        is_meeting = sample_keyword_count >= 3 and total_keyword_count >= 5
        
        return is_meeting, total_keyword_count, total_keywords[:10]

    def extract_entities(self, text):
        """Extract people, dates, and organizations"""
        doc = self.nlp(text)
        entities = {'people': set(), 'dates': set(), 'orgs': set()}
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                entities['people'].add(ent.text)
            elif ent.label_ in ['DATE', 'TIME']:
                entities['dates'].add(ent.text)
            elif ent.label_ == 'ORG':
                entities['orgs'].add(ent.text)
        return entities

    def process_meeting(self, transcribed_text):
        """Process meeting using YOUR trained model"""
        
        print("="*60)
        print("🔍 VALIDATING MEETING CONTENT...")
        print("="*60)
        
        # ✅ VALIDATION: Check if content is actually a meeting
        is_meeting, keyword_count, found_keywords = self.is_meeting_content(transcribed_text)
        
        if not is_meeting:
            raise ValueError(
                f"❌ Sorry, your URL/file doesn't match with office meeting keywords. "
                f"Only {keyword_count} meeting-related keywords found: {', '.join(found_keywords[:5]) if found_keywords else 'none'}. "
                f"Please upload content from a business meeting, conference call, or professional discussion."
            )
        
        print(f"✅ Meeting content validated ({keyword_count} keywords found)")
        print(f"📝 Keywords detected: {', '.join(found_keywords)}")
        
        # Continue with normal processing
        doc = self.nlp(transcribed_text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # Extract entities
        entities = self.extract_entities(transcribed_text)
        
        # Initialize summary
        summary = {
            "Tasks & Action Items": [],
            "Key Decisions & Plans": [],
            "Issues, Risks & Costs": [],
            "Logistics & Schedule": [],
            "Context & Notes": [],
            "Participants": list(entities['people']),
            "Important Dates": list(entities['dates'])
        }
        
        print(f"Processing {len(sentences)} sentences...")
        for sent_str in sentences:
            if not sent_str:
                continue
            lower_sent = sent_str.lower()

            # --- LAYER 1: HARD RULES (Highest Priority) ---
            
            # Task Detection
            task_patterns = [
                r"\b(will|shall|need to|have to|must|should|going to)\s+[\w\s]+(?:by|before|until)",
                r"\b(can you|could you|please|would you|need you to|ask you to|require you to)",
                r"\b(i will|i'll|we will|we'll|he will|she will|they will)\s+(?!be here|be there|arrive)",
                r"\b(responsible for|in charge of|assigned to|take care of|handle|manage)",
                r"\b(action|todo|to do|follow up|complete|deliver|submit|prepare|review)\b"
            ]
            if any(re.search(p, lower_sent) for p in task_patterns):
                summary["Tasks & Action Items"].append(sent_str)
                continue

            # Decision Detection
            decision_patterns = [
                r"\b(decided|agreed|approved|confirmed|concluded|resolved|finalized)",
                r"\b(we agree|consensus|unanimous|voted|motion passed)",
                r"\b(go ahead|proceed with|moving forward|greenlit)"
            ]
            if any(re.search(p, lower_sent) for p in decision_patterns):
                summary["Key Decisions & Plans"].append(sent_str)
                continue

            # Issue Detection
            issue_patterns = [
                r"\b(risk|concern|worry|problem|issue|challenge|obstacle|blocker)",
                r"\b(delay|behind schedule|overrun|exceeded|over budget)",
                r"\b(budget|cost|expense|financial|funding)",
                r"\b(critical|urgent|important|priority|escalate)",
                r"\b(failure|failed|error|mistake|wrong)"
            ]
            if any(re.search(p, lower_sent) for p in issue_patterns):
                summary["Issues, Risks & Costs"].append(sent_str)
                continue

            # Logistics Detection
            logistics_patterns = [
                r"\b(agenda|schedule|timeline|calendar|meeting)",
                r"\b(next meeting|reschedule|postpone|cancel)",
                r"\b(arrive|start|end|break|lunch)",
                r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
                r"\b(january|february|march|april|may|june|july|august|september|october|november|december)",
                r"\d{1,2}:\d{2}|o'clock|\d{1,2}\s*(?:am|pm)"
            ]
            if any(re.search(p, lower_sent) for p in logistics_patterns):
                summary["Logistics & Schedule"].append(sent_str)
                continue

            # Noise Filter
            noise_patterns = [
                r"^(thank you|thanks|okay|alright|sure|yes|no|exactly)[\.,!]?$",
                r"\b(good morning|good afternoon|hello|hi everyone|welcome)",
                r"\b(apologies|sorry i'm late|excuse me)",
                r"\b(is everyone here|everyone is here|we're all here)\b"
            ]
            if any(re.search(p, lower_sent) for p in noise_patterns):
                continue

            # Skip very short sentences
            if len(sent_str.split()) < 5:
                continue

            # --- LAYER 2: YOUR TRAINED MODEL ---
            try:
                clean_tokens = preprocess_text(sent_str)
                clean_str = " ".join(clean_tokens)
                predicted_label = self.model.predict([clean_str])[0]

                # Map to summary category
                category = self.label_mapping.get(predicted_label, "Context & Notes")
                
                if category == "Chit-chat":
                    continue
                else:
                    summary[category].append(sent_str)
                    
            except Exception as e:
                print(f"Warning: Model prediction failed for sentence: {sent_str[:50]}... Error: {e}")
                summary["Context & Notes"].append(sent_str)

        return summary

    def format_summary(self, summary):
        """Generate formatted output"""
        output = []
        output.append("=" * 60)
        output.append("OFFICIAL MEETING MINUTES SUMMARY")
        output.append("=" * 60)

        if summary.get("Participants"):
            output.append("\n### PARTICIPANTS")
            for person in summary["Participants"]:
                output.append(f" - {person}")

        if summary.get("Important Dates"):
            output.append("\n### IMPORTANT DATES & TIMES")
            for date in summary["Important Dates"]:
                output.append(f" - {date}")

        for category, items in summary.items():
            if category not in ["Participants", "Important Dates"] and items:
                output.append(f"\n### {category.upper()}")
                for i, item in enumerate(items, 1):
                    output.append(f" {i}. {item}")

        output.append("\n" + "=" * 60)
        return "\n".join(output)
