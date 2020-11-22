# Care Wheel
With CareWheel, a small-affordable IOT device (depending on your type of supervision) will keep monitoring the user's vitals including - (health related) temperature, heart rate, etc (over 37 parameters) aswell as (general related) location. Moreover, supervisors can upload status update or a picture aswell. All these information are sent to our backend where we use over 20 different statistical techniques to sort the data in more meaningful manner along with Machine Learning and Natural Language Processesing.

# Installation
```bash
# Make sure you have git & pip installed
# Install virtual env
sudo apt-get install python-virtualenv7
# Install Docker & Docker Compose (https://github.com/hellohaptik/chatbot_ner/blob/develop/docs/install.md)

git clone https://github.com/ankitpriyarup/care-wheel.git
cd care-wheel
virtualenv -p python3 env
source env/bin/activate
pip3 install -r requirements.txt

# Install External dependencies
mkdir external && \
cd external && \
git clone https://github.com/hellohaptik/chatbot_ner.git && \
cd chatbot_ner && \
cp config.example .env && \
cp .env docker/.env && \
cd docker && \
docker-compose up --build -d && \
cd ../../../

# nltk.download('stopwords')
# Finally Execute the programm
python3 app.py

# In new terminal, deploy to home server, by exposing local port (install ngrok & cd to that directory)
./ngrok http 5000

# Setup Twilio for whatsapp messaging, open my https://www.twilio.com/ account
# (email: ankitpriyarup@gmail.com password:CareWheelProject)
# Under Programmable SMS > Whatsapp (first send the msg to the whatsapp number mentioned on the page as written)
# Finally set WHEN A MESSAGE COMES IN to the ngrok address you recieved before

# (Optional) To reset database
rm -rf site.db
python3
from app import db
db.create_all()
python3 test/db.py
```

# API
TODO

# TODO
- Redesign frontend UI/UX, make it responsive and more complete website
- Add different error constraints - User with same email already present, hash keys, key complexity and stuffs
- Generalize bunch of redundant, duplicate code
- Improve natural language skills and simmilarity matching method
- Add more non-health parameters currently only location based (latitude & longitude are there)
- Add login & registeration frontend support, make full-blown website
- Upgrade supervisor role, supervisor's should have custom profile page with comments and ratings support
- Create caretaker review system more robust by adding parameter for the person who posted review, checking constraints to avoid multiple review by someone to certain profile
- Dashboard path system not implemented yet
- Dashboard remind medicine not implemented yet
- Implement post creation feature from dashboard
- Implement more features at dashboard side
- Implement update_statistics_shell script
- Implement update_sentiments shell script
- Implement update_ml shell script

# License
Entire project is distributed under the terms of the MIT license, as described in the LICENSE.md file. All rights goes to TechnoNerdz
