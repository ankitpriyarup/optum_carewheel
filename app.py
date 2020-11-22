from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_restful import Resource, Api
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import random
import string
import os
import json
import nltk

app = Flask(__name__, static_url_path='')
api = Api(app)
account_sid = 'ACc81bc8dfc81a1536208ee34d767cf048'
auth_token = 'bb411a9c5f6631634a342779b28cc612'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)
client = Client(account_sid, auth_token)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    fullname = db.Column(db.String(32), unique=False, nullable=False)
    mobile = db.Column(db.String(120), unique=True, nullable=False)
    guardianmobile = db.Column(db.String(120), unique=True, nullable=False)
    publickey = db.Column(db.String(120), nullable=False)
    privatekey = db.Column(db.String(120), nullable=False)
    description = db.Column(db.String(256), unique=False, nullable=True)
    recovery = db.Column(db.Integer, unique=False, nullable=True, default=100)
    mood = db.Column(db.Integer, unique=False, nullable=True, default=100)
    posts = db.relationship('Post', backref='author', lazy=True)
    parameters = db.relationship('Parameters', backref='author', lazy=True)
    picture = db.Column(db.String(120), nullable=True,
                        default='http://www.myiconfinder.com/icon/download/0cb3cbfc32eb4055b95b2f1c0ddd30ff-user.png')

    def __repr__(self):
        return f"User('{self.username}', '{self.mobile}', '{self.publickey}')"


class Supervisor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    fullname = db.Column(db.String(32), unique=False, nullable=False)
    userAssosiated = db.Column(db.String(32), unique=True, nullable=False)
    description = db.Column(db.String(256), unique=False, nullable=True)
    privatekey = db.Column(db.String(120), nullable=False)
    reviews = db.relationship('Review', backref='author', lazy=True)
    picture = db.Column(db.String(120), nullable=True,
                        default='http://www.myiconfinder.com/icon/download/0cb3cbfc32eb4055b95b2f1c0ddd30ff-user.png')
    totalRate = db.Column(db.Float, nullable=True, default=5.0)
    totalCount = db.Column(db.Integer, nullable=True, default=1)

    def __repr__(self):
        return f"Supervisor('{self.username}')"


class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    rate = db.Column(db.Integer, nullable=False)
    title = db.Column(db.String(100), nullable=True)
    date_posted = db.Column(db.DateTime, nullable=False,
                            default=datetime.utcnow)
    content = db.Column(db.Text, nullable=True)
    supervisor_id = db.Column(db.Integer, db.ForeignKey(
        'supervisor.id'), nullable=False)

    def __repr__(self):
        return f"Review('{self.title}', '{self.content}', '{self.date_posted}')"


class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    date_posted = db.Column(db.DateTime, nullable=False,
                            default=datetime.utcnow)
    content = db.Column(db.Text, nullable=False)
    media = db.Column(db.Text, nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f"Post('{self.title}', '{self.content}', '{self.date_posted}')"


class Parameters(db.Model):
    p_albumin = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_dist = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_alp = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_alt = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_ast = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_bilirubin = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_bun = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_cholestrol = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_diasabp = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_fio2 = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_gcs = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_glucose = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_hco3 = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_hct = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_hr = db.Column(db.Integer, unique=False, nullable=True, default=-1)
    p_k = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_lactate = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_mg = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_map = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_mechvent = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_na = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_nidiasabp = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_nimap = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_nisysabp = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_paco2 = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_pao2 = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_ph = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_platelets = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_resprate = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_sao2 = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_sysabp = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_temp = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_tropl = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_tropt = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_urine = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_wbc = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_weight = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_height = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_age = db.Column(db.Integer, unique=False, nullable=True, default=-1)
    p_lat = db.Column(db.Float, unique=False, nullable=True, default=-1)
    p_long = db.Column(db.Float, unique=False, nullable=True, default=-1)
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date_posted = db.Column(db.DateTime, nullable=False,
                            default=datetime.utcnow)

    def __repr__(self):
        return f"Parameter('{self.user_id}', '{self.date_posted}')"


GREETING_INPUTS = ("hello", "helloo", "hellooo", "helloooo", "hi", "hii", "hiii", "hiiii", "greetings",
                   "sup", "what's up", "hey", "heyy", "heyyy", "heyyyy", "heyyyyy", "namaste", "hello ji")
GREETING_RESPONSES = ["hi", "hey", "*nods*",
                      "hi there", "hello", "I'm here to help"]
COMMON_INPUTS = ("thanks", "thankss", "thanksss", "thankssss", "good", "goood", "gooood",
                 "great", "perfect", "thank you", "thank youu", "thank youuu", "dhanyawaad", "shukriyaa")
COMMON_RESPONSES = [":)", "Glad to hear that!", ":D"]
REPORT = ("show", "statistics", "data", "report", "analysis", "status")
CURRENT = ("cur", "current", "recent", "now", "live", "abhi",
           "abhi", "turant wali", "turant", "eesi waqt")


def match_sentence_resp(sentence, match_inp, match_out):
    return random.choice(match_out)


def match_sentence(sentence, match_inp):
    score = 0.0
    for word in match_inp:
        X_list = word_tokenize(sentence.lower())
        Y_list = word_tokenize(word.lower())
        sw = stopwords.words('english')
        l1 = []
        l2 = []
        X_set = {w for w in X_list if not w in sw}
        Y_set = {w for w in Y_list if not w in sw}
        rvector = X_set.union(Y_set)
        for w in rvector:
            if w in X_set:
                l1.append(1)
            else:
                l1.append(0)
            if w in Y_set:
                l2.append(1)
            else:
                l2.append(0)
        c = 0
        for i in range(len(rvector)):
            c += l1[i]*l2[i]
        den = float((sum(l1)*sum(l2))**0.5)
        if den != 0:
            cosine = c / den
            score = score + cosine
    return score > 0.3


def ner_process(sentence):
    open('commands/_i', 'w').close()
    f = open("commands/_i", "a")
    f.write(sentence)
    f.close()
    os.system('sh ./commands/ner_time.sh')
    json_res = {}
    with open('commands/_o') as fp:
        line = fp.readline()
        while line:
            line = fp.readline()
            line = line.replace('HTTP/1.1 200 OK', '')
            if 'data' in line and 'null' not in line:
                json_res.update(json.loads(line))
    return json_res


@app.route('/data/<path:path>')
def res(path):
    return send_from_directory('data', path)


@app.route("/chat_reply", methods=['POST'])
def chat_reply():
    sender = request.form.get('From')
    sender = sender.replace('whatsapp:', '')
    msg = request.form.get('Body')
    resp = MessagingResponse()
    raw = str(msg).lower()

    if (match_sentence(raw, GREETING_INPUTS)):
        resp.message(match_sentence_resp(
            raw, GREETING_INPUTS, GREETING_RESPONSES))
    elif (match_sentence(raw, COMMON_INPUTS)):
        resp.message(match_sentence_resp(raw, COMMON_INPUTS, COMMON_RESPONSES))
    elif (match_sentence(raw, REPORT)):
        user = User.query.filter_by(guardianmobile=sender).first()
        if user:
            if (match_sentence(raw, CURRENT)):
                param = user.parameters[len(user.parameters)-1]
                msg = ''
                if param.p_dist != -1:
                    msg += 'Ultrasonic Distance: ' + str(param.p_dist)
                if param.p_albumin != -1:
                    msg += 'Albumin: ' + str(param.p_albumin) + ' g/dL\n'
                if param.p_alp != -1:
                    msg += 'Alkaline Phosphate: ' + \
                        str(param.p_alp) + ' IU/L\n'
                if param.p_alt != -1:
                    msg += 'Alanine transaminase: ' + \
                        str(param.p_alt) + ' IU/L\n'
                if param.p_ast != -1:
                    msg += 'Aspartate transaminase: ' + \
                        str(param.p_ast) + ' IU/L\n'
                if param.p_bilirubin != -1:
                    msg += 'Bilirubin: ' + str(param.p_bilirubin) + ' mg/dL\n'
                if param.p_bun != -1:
                    msg += 'Blood Urea Nitrogen: ' + \
                        str(param.p_bun) + ' mg/dL\n'
                if param.p_cholestrol != -1:
                    msg += 'Cholestrol: ' + \
                        str(param.p_cholestrol) + ' mg/dL\n'
                if param.p_diasabp != -1:
                    msg += 'Invasive diastolic arterial blood pressure: ' + \
                        str(param.p_diasabp) + ' mmHg\n'
                if param.p_fio2 != -1:
                    msg += 'Fractional inspired O2: ' + \
                        str(param.p_fio2) + ' (0-1)\n'
                if param.p_gcs != -1:
                    msg += 'Glasgow Coma Score: ' + \
                        str(param.p_gcs) + ' (3-15)\n'
                if param.p_glucose != -1:
                    msg += 'Glucose: ' + str(param.p_glucose) + ' mg/dL\n'
                if param.p_hco3 != -1:
                    msg += 'Bicarbonate: ' + str(param.p_hco3) + ' mmol/L\n'
                if param.p_hct != -1:
                    msg += 'Hematocrit: ' + str(param.p_hct) + ' %\n'
                if param.p_hr != -1:
                    msg += 'Heart Rate: ' + str(param.p_hr) + ' bpm\n'
                if param.p_k != -1:
                    msg += 'Potassium: ' + str(param.p_k) + ' mEq/L\n'
                if param.p_lactate != -1:
                    msg += 'Lactate: ' + str(param.p_lactate) + ' mmol/L\n'
                if param.p_mg != -1:
                    msg += 'Magnesium: ' + str(param.p_mg) + ' mmol/L\n'
                if param.p_map != -1:
                    msg += 'MAP: ' + str(param.p_map) + ' mmHg\n'
                if param.p_mechvent != -1:
                    msg += 'Mechanical ventilation respiration: ' + \
                        str(param.p_mechvent) + ' (0:false, or 1:true)\n'
                if param.p_na != -1:
                    msg += 'Sodium: ' + str(param.p_na) + ' mEq/L\n'
                if param.p_nidiasabp != -1:
                    msg += 'Non-invasive diastolic arterial blood pressure: ' + \
                        str(param.p_nidiasabp) + ' mmHg\n'
                if param.p_nimap != -1:
                    msg += 'Non-invasive mean arterial blood pressure: ' + \
                        str(param.p_nimap) + ' mmHg\n'
                if param.p_nisysabp != -1:
                    msg += 'Non-invasive systolic arterial blood pressure: ' + \
                        str(param.p_nisysabp) + ' mmHg\n'
                if param.p_paco2 != -1:
                    msg += 'partial pressure of arterial CO2: ' + \
                        str(param.p_paco2) + ' mmHg\n'
                if param.p_pao2 != -1:
                    msg += 'Partial pressure of arterial O2: ' + \
                        str(param.p_pao2) + ' mmHg\n'
                if param.p_ph != -1:
                    msg += 'Arterial pH: ' + str(param.p_ph) + ' (0-14)\n'
                if param.p_platelets != -1:
                    msg += 'Platelets: ' + \
                        str(param.p_platelets) + ' cells/nL\n'
                if param.p_resprate != -1:
                    msg += 'Respiration rate: ' + \
                        str(param.p_resprate) + ' bpm\n'
                if param.p_sao2 != -1:
                    msg += 'O2 saturation in hemoglobin: ' + \
                        str(param.p_sao2) + ' %\n'
                if param.p_sysabp != -1:
                    msg += 'Invasive systolic arterial blood pressure: ' + \
                        str(param.p_sysabp) + ' mmHg\n'
                if param.p_temp != -1:
                    msg += 'Temperature: ' + str(param.p_temp) + ' °C\n'
                if param.p_tropl != -1:
                    msg += 'Troponin-I: ' + str(param.p_tropl) + ' μg/L\n'
                if param.p_tropt != -1:
                    msg += 'Troponin-T: ' + str(param.p_tropt) + ' μg/L\n'
                if param.p_urine != -1:
                    msg += 'Urine output: ' + str(param.p_urine) + ' mL\n'
                if param.p_wbc != -1:
                    msg += 'White blood cell count: ' + \
                        str(param.p_wbc) + ' cells/nL\n'
                if param.p_weight != -1:
                    msg += 'Weight: ' + str(param.p_weight) + ' kg\n'
                if param.p_height != -1:
                    msg += 'Height: ' + str(param.p_height) + ' cm\n'

                if msg == '':
                    resp.message(
                        "I'm afraid, nothing has been recorded yet :(")
                else:
                    resp.message(
                        "*Here's the current report you requested-*\n" + msg)
            else:
                json_res = ner_process(raw)
                present = False
                for item in json_res['data']:
                    present = True
                    if item['entity_value']['value']:
                        resp.message("Here's the report from " + str(
                            item['entity_value']['value']) + "\nhttps://care-wheel.herokuapp.com/user/" + user.username)
                if present is False:
                    resp.message(
                        "Here's the complete report you requested-\nhttps://care-wheel.herokuapp.com/user/" + user.username)
        else:
            resp.message(
                "Sorry you are not registered as anyone's guardian yet! Please install our mobile app to get started :)")
    else:
        resp.message("I'm sorry! I couldn't get that")

    return str(resp)

# Required (username, fullname, mobile, publickey, privatekey, guardianmobile)
# Optional (description, picture)
@app.route('/api/assign_user/<uuid>', methods=['GET', 'POST'])
def assign_user(uuid):
    content = request.json
    if uuid != auth_token:
        return jsonify({"output": "Access Denied!"})
    else:
        user = User(username=content['username'], fullname=content['fullname'], mobile=content['mobile'],
                    publickey=content['publickey'], privatekey=content['privatekey'], guardianmobile=content['guardianmobile'])
        if 'description' in content:
            user.description = content['description']
        if 'picture' in content:
            user.picture = content['picture']
        db.session.add(user)
        db.session.commit()
        user = User.query.filter_by(username=content['username']).first()
        update = Parameters(user_id=user.id)
        db.session.add(update)
        db.session.commit()

        message = client.messages \
            .create(
                from_='whatsapp:+14155238886',
                body="Welcome to Care Wheel. Now you're just a tap away from your loved ones <3\nYou can always rely on me for frequent updates ;)",
                to='whatsapp:' + user.mobile
            )

        return jsonify({"output": "User successfully created!"})

# Required (username, fullname, privatekey)
# Optional (description, picture)
@app.route('/api/assign_supervisor/<uuid>', methods=['GET', 'POST'])
def assign_supervisor(uuid):
    content = request.json
    if uuid != auth_token:
        return jsonify({"output": "Access Denied!"})
    else:
        supervisor = Supervisor(username=content['username'], fullname=content['fullname'],
                                userAssosiated=content['userAssosiated'], privatekey=content['privatekey'])
        if 'description' in content:
            supervisor.description = content['description']
        if 'picture' in content:
            supervisor.picture = content['picture']
        db.session.add(supervisor)
        db.session.commit()
        return jsonify({"output": "Supervisor successfully created!"})

# Required (username, key, title, msg) optional (media)
@app.route('/api/post_task/<uuid>', methods=['GET', 'POST'])
def post_task(uuid):
    content = request.json
    if uuid != auth_token:
        return jsonify({"output": "Access Denied!"})
    else:
        user = User.query.filter_by(username=content['username']).first()
        if user:
            if user.publickey == content['key']:
                post = Post(title=content['title'],
                            content=content['msg'], user_id=user.id)
                db.session.add(post)
                if content['title'] == ' ' or content['title'] == '':
                    msg = '*[updates]*' + '\n\n' + content['msg']
                else:
                    msg = '*[updates] ' + content['title'] + \
                        '*' + '\n\n' + content['msg']
                if 'media' in content:
                    post.media = content['media']
                    message = client.messages \
                        .create(
                            media_url=[content['media']],
                            from_='whatsapp:+14155238886',
                            body=msg,
                            to='whatsapp:' + user.mobile
                        )
                else:
                    message = client.messages \
                        .create(
                            from_='whatsapp:+14155238886',
                            body=msg,
                            to='whatsapp:' + user.mobile
                        )
                db.session.commit()
                return jsonify({"output": "Task successfully recorded!"})
            else:
                return jsonify({"output": "User not authorized!"})
        else:
            return jsonify({"output": "User not found!"})

@app.route('/api/post_msg/<content>', methods=['GET', 'POST'])
def send_msg(content):
    print("SENDING")
    message = client.messages \
            .create(
                from_='whatsapp:+14155238886',
                body=content,
                to='whatsapp:+919818865785'
            )

# required (username, rate - between [1 to 5]) optional (title, content)
@app.route('/api/post_review/<uuid>', methods=['GET', 'POST'])
def post_review(uuid):
    content = request.json
    if uuid != auth_token:
        return jsonify({"output": "Access Denied!"})
    else:
        supervisor = Supervisor.query.filter_by(
            username=content['username']).first()
        if supervisor:
            if content['rate'] > 5 or content['rate'] < 1:
                return jsonify({"output": "Invalid input!"})
            else:
                review = Review(supervisor_id=supervisor.id,
                                rate=content['rate'])
                if 'title' in content:
                    review.title = content['title']
                if 'content' in content:
                    review.content = content['content']
                db.session.add(review)
                supervisor.totalRate = (supervisor.totalRate + review.rate) / 2
                supervisor.totalCount = supervisor.totalCount + 1
                db.session.commit()
                return jsonify({"output": "Review successfully posted!"})
        else:
            return jsonify({"output": "User not authorized!"})

# required (username, key) optional (key value pair of properties)
@app.route('/api/post_update/<uuid>', methods=['GET', 'POST'])
def post_update(uuid):
    content = request.json
    if uuid != auth_token:
        return jsonify({"output": "Access Denied!"})
    else:
        user = User.query.filter_by(username=content['username']).first()
        if user:
            if user.publickey == content['key']:
                param = user.parameters[len(user.parameters)-1]
                update = Parameters(user_id=user.id)
                update.p_albumin = param.p_albumin
                update.p_dist = param.p_dist
                update.p_alp = param.p_alp
                update.p_alt = param.p_alt
                update.p_ast = param.p_ast
                update.p_bilirubin = param.p_bilirubin
                update.p_bun = param.p_bun
                update.p_cholestrol = param.p_cholestrol
                update.p_diasabp = param.p_diasabp
                update.p_fio2 = param.p_fio2
                update.p_gcs = param.p_gcs
                update.p_glucose = param.p_glucose
                update.p_hco3 = param.p_hco3
                update.p_hct = param.p_hct
                update.p_hr = param.p_hr
                update.p_k = param.p_k
                update.p_lactate = param.p_lactate
                update.p_mg = param.p_mg
                update.p_map = param.p_map
                update.p_mechvent = param.p_mechvent
                update.p_na = param.p_na
                update.p_nidiasabp = param.p_nidiasabp
                update.p_nimap = param.p_nimap
                update.p_nisysabp = param.p_nisysabp
                update.p_paco2 = param.p_paco2
                update.p_pao2 = param.p_pao2
                update.p_ph = param.p_ph
                update.p_platelets = param.p_platelets
                update.p_resprate = param.p_resprate
                update.p_sao2 = param.p_sao2
                update.p_sysabp = param.p_sysabp
                update.p_temp = param.p_temp
                update.p_tropl = param.p_tropl
                update.p_tropt = param.p_tropt
                update.p_urine = param.p_urine
                update.p_wbc = param.p_wbc
                update.p_weight = param.p_weight
                update.p_height = param.p_height
                update.p_age = param.p_age
                update.p_lat = param.p_lat
                update.p_long = param.p_long

                for key, value in content.items():
                    if key == 'distance':
                        update.p_dist = value
                    if key == 'albumin':
                        update.p_albumin = value
                    if key == 'alp':
                        update.p_alp = value
                    if key == 'alt':
                        update.p_alt = value
                    if key == 'ast':
                        update.p_ast = value
                    if key == 'bilirubin':
                        update.p_bilirubin = value
                    if key == 'bun':
                        update.p_bun = value
                    if key == 'cholestrol':
                        update.p_cholestrol = value
                    if key == 'diasabp':
                        update.p_diasabp = value
                    if key == 'fio2':
                        update.p_fio2 = value
                    if key == 'gcs':
                        update.p_gcs = value
                    if key == 'glucose':
                        update.p_glucose = value
                    if key == 'hco3':
                        update.p_hco3 = value
                    if key == 'hct':
                        update.p_hct = value
                    if key == 'hr':
                        update.p_hr = value
                    if key == 'k':
                        update.p_k = value
                    if key == 'lactate':
                        update.p_lactate = value
                    if key == 'mg':
                        update.p_mg = value
                    if key == 'map':
                        update.p_map = value
                    if key == 'mechvent':
                        update.p_mechvent = value
                    if key == 'na':
                        update.p_na = value
                    if key == 'nidiasabp':
                        update.p_nidiasabp = value
                    if key == 'nimap':
                        update.p_nimap = value
                    if key == 'nisysabp':
                        update.p_nisysabp = value
                    if key == 'paco2':
                        update.p_paco2 = value
                    if key == 'pao2':
                        update.p_pao2 = value
                    if key == 'ph':
                        update.p_ph = value
                    if key == 'platelets':
                        update.p_platelets = value
                    if key == 'resprate':
                        update.p_resprate = value
                    if key == 'sao2':
                        update.p_sao2 = value
                    if key == 'sysabp':
                        update.p_sysabp = value
                    if key == 'temp':
                        update.p_temp = value
                    if key == 'tropl':
                        update.p_tropl = value
                    if key == 'tropt':
                        update.p_tropt = value
                    if key == 'urine':
                        update.p_urine = value
                    if key == 'wbc':
                        update.p_wbc = value
                    if key == 'weight':
                        update.p_weight = value
                    if key == 'height':
                        update.p_height = value
                    if key == 'age':
                        update.p_age = value
                    if key == 'lattitude':
                        update.p_lat = value
                    if key == 'longitude':
                        update.p_long = value
                db.session.add(update)
                db.session.commit()
                print("Post Updated successfully")
                print(param)
                return jsonify({"output": "Post updated successfully!"})
            else:
                return jsonify({"output": "User not authorized!"})
        else:
            return jsonify({"output": "User not found!"})


# Frontend part
@app.route("/user/<string:_username>")
def user(_username):
    user = User.query.filter_by(username=_username).first()
    if user:
        param = user.parameters[len(user.parameters)-1]
        _posts = user.posts
        return render_template('user_profile.html', username=_username, name=user.fullname, desc=user.description, mobile=user.mobile,
                               gmobile=user.guardianmobile, pic=user.picture, recovery=user.recovery, mood=user.mood, parameter=param, posts=_posts)
    else:
        return 'User not found!'


@app.route("/caretaker/<string:_username>")
def caretaker(_username):
    caretaker = Supervisor.query.filter_by(username=_username).first()
    if caretaker:
        _reviews = caretaker.reviews
        return render_template('user_caretaker.html', username=_username, name=caretaker.fullname, desc=caretaker.description, pic=caretaker.picture,
                               reviews=_reviews, totalRate=caretaker.totalRate, totalCount=caretaker.totalCount)
    else:
        return 'Caretaker not found!'


@app.route("/dashboard/<string:_username>")
def dashboard(_username):
    caretaker = Supervisor.query.filter_by(username=_username).first()
    if caretaker:
        return render_template('dashboard.html', caretaker=_username, name=caretaker.fullname, user=caretaker.userAssosiated)
    else:
        return 'Caretaker not found'


if __name__ == "__main__":
    app.run(host='127.0.0.1', debug=True)
