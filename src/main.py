# Main entry point for Training Session Monitoring System


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from datetime import datetime
from utils.email_sender import send_email_reminder

TRAINING_LIST_PATH = '/home/rachana/Desktop/team_management/training_monitor_system/training_list.xlsx'


def get_upcoming_sessions(file_path, num_sessions=4):
    df = pd.read_excel(file_path)
    today = datetime.now().date()
    df['date'] = pd.to_datetime(df['date']).dt.date
    # Select sessions with date >= today, sort by date, and pick the next 4
    upcoming_sessions = df[df['date'] >= today].sort_values('date').head(num_sessions)
    return upcoming_sessions


def main():
    sessions = get_upcoming_sessions(TRAINING_LIST_PATH)
    for idx, (_, session) in enumerate(sessions.iterrows()):
        trainer_email = session['email']
        session_name = session['title']
        cc_email = 'jaimin.m@simformsolutions.com' if idx == 0 else None
        send_email_reminder(trainer_email, session_name, session, cc_email=cc_email)
    print(f"Sent reminders for {len(sessions)} sessions.")


if __name__ == "__main__":
    main()
