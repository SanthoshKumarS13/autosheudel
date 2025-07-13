# state_manager.py
import json
import os
from datetime import datetime, timedelta, UTC # NEW: Import datetime, timedelta, UTC
from config import CONTENT_TYPE_CYCLE, JSON_OUTPUT_DIR, WEEKLY_ANALYSIS_INTERVAL_DAYS # Import WEEKLY_ANALYSIS_INTERVAL_DAYS

class WorkflowStateManager:
    """Manages the state of the content generation workflow."""

    STATE_FILE = f"{JSON_OUTPUT_DIR}/state.json"

    def __init__(self):
        self.current_post_type_index = 0
        self.posts_generated_in_cycle = 0 # Tracks how many posts generated since last full cycle reset
        self.last_analysis_timestamp = None # NEW: Timestamp for last weekly performance analysis
        self._load_state()

    def _load_state(self):
        """Loads the last saved state from a JSON file."""
        os.makedirs(os.path.dirname(self.STATE_FILE), exist_ok=True) # Ensure directory exists
        if os.path.exists(self.STATE_FILE):
            try:
                with open(self.STATE_FILE, 'r') as f:
                    state = json.load(f)
                    self.current_post_type_index = state.get('current_post_type_index', 0)
                    self.posts_generated_in_cycle = state.get('posts_generated_in_cycle', 0)
                    # NEW: Load last_analysis_timestamp
                    last_ts_str = state.get('last_analysis_timestamp')
                    if last_ts_str:
                        self.last_analysis_timestamp = datetime.fromisoformat(last_ts_str).replace(tzinfo=UTC)
                    else:
                        self.last_analysis_timestamp = datetime.now(UTC) # Initialize if not found
                print(f"State loaded: current_post_type_index={self.current_post_type_index}, posts_generated_in_cycle={self.posts_generated_in_cycle}, last_analysis_timestamp={self.last_analysis_timestamp}")
            except json.JSONDecodeError:
                print(f"State file {self.STATE_FILE} is corrupted. Initializing new state.")
                self._save_state() # Save a fresh state
            except Exception as e: # Catch other potential errors during loading/parsing
                print(f"Error loading state file {self.STATE_FILE}: {e}. Initializing new state.")
                self._save_state()
        else:
            print(f"State file {self.STATE_FILE} not found. Initializing new state.")
            self.last_analysis_timestamp = datetime.now(UTC) # Initialize on first run
            self._save_state() # Save a fresh state

    def _save_state(self):
        """Saves the current state to a JSON file."""
        state = {
            'current_post_type_index': self.current_post_type_index,
            'posts_generated_in_cycle': self.posts_generated_in_cycle,
            'last_analysis_timestamp': self.last_analysis_timestamp.isoformat() if self.last_analysis_timestamp else None # NEW: Save timestamp as ISO format string
        }
        os.makedirs(os.path.dirname(self.STATE_FILE), exist_ok=True) # Ensure directory exists
        with open(self.STATE_FILE, 'w') as f:
            json.dump(state, f, indent=4)
        print(f"State saved: current_post_type_index={self.current_post_type_index}, posts_generated_in_cycle={self.posts_generated_in_cycle}, last_analysis_timestamp={self.last_analysis_timestamp}")

    def get_current_post_type(self):
        """Returns the content type for the current post in the cycle."""
        return CONTENT_TYPE_CYCLE[self.current_post_type_index % len(CONTENT_TYPE_CYCLE)]

    def get_current_post_number(self):
        """Returns the sequential post number within the current cycle."""
        return (self.posts_generated_in_cycle % len(CONTENT_TYPE_CYCLE)) + 1

    def increment_post_type_index(self):
        """Increments the index for the content type cycle and updates generated posts count."""
        self.current_post_type_index += 1
        self.posts_generated_in_cycle += 1
        self._save_state()

    def should_run_weekly_analysis(self):
        """Checks if a week has passed since the last performance analysis."""
        if not self.last_analysis_timestamp:
            print("No last analysis timestamp found, running analysis for the first time.")
            return True # Run on first execution if timestamp is not set

        time_since_last_analysis = datetime.now(UTC) - self.last_analysis_timestamp
        if time_since_last_analysis >= timedelta(days=WEEKLY_ANALYSIS_INTERVAL_DAYS):
            print(f"It has been {time_since_last_analysis.days} days since last analysis. Running weekly analysis.")
            return True
        else:
            days_left = WEEKLY_ANALYSIS_INTERVAL_DAYS - time_since_last_analysis.days
            print(f"Weekly analysis not due yet. {days_left} days remaining until next analysis.")
            return False

    def update_last_analysis_timestamp(self):
        """Updates the last analysis timestamp to the current time."""
        self.last_analysis_timestamp = datetime.now(UTC)
        self._save_state()
