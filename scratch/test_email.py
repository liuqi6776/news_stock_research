import os
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv

# Load env variables
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

def test_send():
    mail_host = os.getenv("SMTP_SERVER", "smtp.163.com")
    mail_user = os.getenv("SMTP_USER", "13259770650@163.com")
    mail_pass = os.getenv("SMTP_PASSWORD")
    smtp_port = int(os.getenv("SMTP_PORT", "465"))
    
    sender = mail_user
    receivers = ["568701293@qq.com"]
    
    print(f"Connecting to {mail_host}:{smtp_port} using SSL...")
    print(f"Login User: {mail_user}")
    
    message = MIMEText("<p>This is a test email from your Antigravity Quant trading system to verify SMTP SSL connection.</p>", "html", "utf-8")
    message["Subject"] = "Antigravity Quant System - Connection Test"
    message["From"] = sender
    message["To"] = receivers[0]
    
    try:
        # Connect via SMTP_SSL (Port 465)
        server = smtplib.SMTP_SSL(mail_host, smtp_port, timeout=10)
        print("Connected! Logging in...")
        
        # We can try logging in with the short username
        login_user = "13259770650"
        print(f"Attempting login as: {login_user}")
        server.login(login_user, mail_pass)
        
        print("Logged in successfully! Sending email...")
        server.sendmail(sender, receivers, message.as_string())
        server.quit()
        print("[SUCCESS] Test email sent successfully to 568701293@qq.com!")
    except Exception as e:
        print(f"[ERROR] Failed to send test email: {e}")

if __name__ == "__main__":
    test_send()
