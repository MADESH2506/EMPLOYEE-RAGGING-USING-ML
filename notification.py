import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import os
import warnings
warnings.filterwarnings("ignore")
import cv2
import HandDataCollecter
import mediapipe as mp
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from geopy.geocoders import Nominatim

loc = Nominatim(user_agent="GetLoc")
from cv2 import imwrite
# entering the location name
import cv2
import HandDataCollecter
import mediapipe as mp
import numpy as np
########Initialise random forest

local_path = (os.path.dirname(os.path.realpath('__file__')))

file_name = ('data.csv')  # file of total data
data_path = os.path.join(local_path, file_name)
print(data_path)
df = pd.read_csv(r'' + data_path)

print(df)

units_in_data = 28  # no. of units in data

titles = []
for i in range(units_in_data):
    titles.append("unit-" + str(i))
X = df[titles]
y = df['letter']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

clf = RandomForestClassifier(n_estimators=30)  # random forest
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
cmrf = confusion_matrix(y_test, y_pred)
print("1.Random Forest Accuracy")

print("Random Forest classification_report")
print(classification_report(y_pred, y_test, labels=None))
print("Random Forest confusion_matrix")


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Initialize individual models
clf_rf = RandomForestClassifier(n_estimators=30)
clf_knn = KNeighborsClassifier()
clf_svc = SVC()

# Create a Stacking Classifier with a meta-model (Logistic Regression)
stacking_clf = StackingClassifier(estimators=[
    ('random_forest', clf_rf),
    ('knn', clf_knn),
    ('svm', clf_svc)
], final_estimator=LogisticRegression())

# Fit the stacking model on the training data
stacking_clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred_stacking = stacking_clf.predict(X_test)




#########Begin predictions
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def get_prediction(image):
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        ImageData = HandDataCollecter.ImageToDistanceData(image, hands)
        DistanceData = ImageData['Distance-Data']
        image = ImageData['image']
        prediction = clf.predict([DistanceData])
        return prediction[0]
def send_email(image, subject, body):
    # Save the image as an attachment
    imwrite("img.png", image)
    
    # Email addresses and login credentials
    fromaddr = "madesh06k@gmail.com"
    toaddr = "madesh06k@gmail.com"
    password = "jues hkqw xsjo rcaw"  # Use an app-specific password 

    # Create the MIMEMultipart message
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = subject

    # HTML structure for the email body
    html_content = f"""
    <html>
        <head>
            <style>
                .alert {{
                    background-color: #ffcccb;
                    color: #a94442;
                    padding: 15px;
                    font-weight: bold;
                    border-radius: 5px;
                    border: 1px solid #f5c6cb;
                    text-align: center;
                    font-size: 18px;
                    margin-bottom: 20px;
                }}
                .content {{
                    font-family: Arial, sans-serif;
                    font-size: 16px;
                    line-height: 1.6;
                    color: #333;
                }}
                .footer {{
                    margin-top: 20px;
                    font-size: 12px;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <div class="alert">🚨 Attention Required</div>
            <div class="content">
                <p>{body}</p>
                <p>Attached is an image related to this alert for your reference.</p>
            </div>
            <div class="footer">
                <p>This email was sent automatically. If you need further assistance, please reply to this message.</p>
            </div>
        </body>
    </html>
    """
    # Attach HTML content
    msg.attach(MIMEText(html_content, 'html'))

    # Attach the image
    filename = "img.png"
    with open(filename, "rb") as attachment:
        mime_base = MIMEBase('application', 'octet-stream')
        mime_base.set_payload(attachment.read())
        encoders.encode_base64(mime_base)
        mime_base.add_header('Content-Disposition', f"attachment; filename= {filename}")
        msg.attach(mime_base)

    # Send the email
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(fromaddr, password)
            server.sendmail(fromaddr, toaddr, msg.as_string())
        print("Mail sent successfully.")
    except smtplib.SMTPException as e:
        print(f"Error sending mail: {e}")


