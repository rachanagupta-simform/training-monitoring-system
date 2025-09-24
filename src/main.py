# Main entry point for Training Session Monitoring System  # High-level description of this script


import sys  # Provides access to system-specific parameters and functions
import os  # OS module for path manipulations
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add project root to Python path for module imports
import pandas as pd  # Pandas for reading and processing the Excel file
from datetime import datetime  # For working with current date/time
from utils.email_sender import send_email_reminder  # Custom function to send reminder emails

TRAINING_LIST_PATH = '/home/rachana/Desktop/team_management/training_monitor_system/training_list.xlsx'  # Absolute path to the Excel file containing training sessions


def get_upcoming_sessions(file_path, num_sessions=4):  # Retrieve the next upcoming training sessions
    df = pd.read_excel(file_path)  # Read the Excel file into a DataFrame
    today = datetime.now().date()  # Get today's date (no time component)
    df['date'] = pd.to_datetime(df['date']).dt.date  # Normalize the 'date' column to date objects
    # Select sessions with date >= today, sort by date, and pick the specified number
    upcoming_sessions = df[df['date'] >= today].sort_values('date').head(num_sessions)  # Filter, sort, and limit rows
    return upcoming_sessions  # Return the filtered DataFrame


def main():  # Main workflow for sending reminders
    sessions = get_upcoming_sessions(TRAINING_LIST_PATH)  # Fetch the upcoming sessions from Excel
    for idx, (_, session) in enumerate(sessions.iterrows()):  # Iterate through each session row with index
        trainer_email = session['email']  # Extract trainer email from the row
        session_name = session['title']  # Extract session title from the row
        cc_email = 'jaimin.m@simformsolutions.com' if idx == 0 else None  # CC first email only (optional logic)
        send_email_reminder(trainer_email, session_name, session, cc_email=cc_email)  # Trigger the reminder email
    print(f"Sent reminders for {len(sessions)} sessions.")  # Log how many reminders were sent


if __name__ == "__main__":  # Ensure script runs only when executed directly
    main()  # Call the main function
