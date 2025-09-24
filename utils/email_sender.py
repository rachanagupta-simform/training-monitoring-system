import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

SMTP_SERVER = os.getenv('SMTP_SERVER')
SMTP_PORT = int(os.getenv('SMTP_PORT'))
SMTP_USER = os.getenv('SMTP_USER')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')


def send_email_reminder(to_email, session_name, session_row, cc_email=None):
    session_date = session_row['date']
    subject = f"Reminder: Training Session '{session_name}' on {session_date}"
    body = f"""
<html>
<body>
Dear Trainer,<br><br>
This is a kind reminder that you are scheduled to conduct the following training session:<br><br>
<b>Session Title:</b> {session_row['title']}<br>
<b>Presenter:</b> {session_row['presenter']}<br>
<b>Date:</b> {session_date}<br><br>
Please plan your work and leaves accordingly to ensure the session runs smoothly.<br><br>
Thank you.<br><br>
Best Regards,<br>
Dr. Rachana Gupta<br>
Senior Software Engineer<br>
rachana.gupta@simformsolutions.com | +91 7984087858<br> 
</body>
</html>
"""

    msg = MIMEMultipart()
    msg['From'] = SMTP_USER
    msg['To'] = to_email
    msg['Subject'] = subject
    if cc_email:
        msg['Cc'] = cc_email
    msg.attach(MIMEText(body, 'html'))

    recipients = [to_email]
    if cc_email:
        recipients.append(cc_email)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, recipients, msg.as_string())
        print(f"Email sent to {to_email}" + (f" (cc: {cc_email})" if cc_email else ""))
    except Exception as e:
        print(f"Failed to send email to {to_email}: {e}")
