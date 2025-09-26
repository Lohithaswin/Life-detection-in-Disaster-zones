import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Function to send email alert
def send_email_alert(to_email, subject, message_body):
    # Your email credentials
    sender_email = "akshayofficialnew@gmail.com"
    app_password = "yzdk uwwb ttvv jmnr"

    # Creating the email content
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = to_email
    message["Subject"] = subject
    message.attach(MIMEText(message_body, "plain"))

    # Connecting to the Gmail SMTP server
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()  # Secure the connection
        server.login(sender_email, app_password)  # Login to the email account
        server.sendmail(sender_email, to_email, message.as_string())  # Send the email

# Main function to generate coordinates and send the alert
def main():
    try:

        # Alert details
        subject = "Disaster Alert"
        message_body = (
            f"Alert: Humans, animals, or vehicles are detected in a disaster zone.\n\n"
        )

        # Send the email alert
        recipient_email = "akshayofficialnew@gmail.com"  # Replace with your email
        send_email_alert(recipient_email, subject, message_body)
        print("Alert email sent successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the script
if __name__ == "__main__":
    main()