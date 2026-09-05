
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, make_response
import io
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import bcrypt
import joblib
import numpy as np
import json
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))

app = Flask(__name__)
app.config["SECRET_KEY"] = "heartguard-secret-key-2024"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///heartguard.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# Load model and scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

# ─── Database Models ───────────────────────────────────────────

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship("Prediction", backref="user", lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(IST).replace(tzinfo=None))
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    height = db.Column(db.Float)
    weight = db.Column(db.Float)
    bmi = db.Column(db.Float)
    ap_hi = db.Column(db.Integer)
    ap_lo = db.Column(db.Integer)
    cholesterol = db.Column(db.Integer)
    gluc = db.Column(db.Integer)
    smoke = db.Column(db.Integer)
    alco = db.Column(db.Integer)
    active = db.Column(db.Integer)
    risk_percent = db.Column(db.Float)
    risk_level = db.Column(db.String(20))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ─── Routes ────────────────────────────────────────────────────

@app.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for("predict_page"))
    return render_template("landing.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
            login_user(user)
            return redirect(url_for("predict_page"))
        flash("Invalid email or password")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered")
            return redirect(url_for("register"))
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        new_user = User(name=name, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for("predict_page"))
    return render_template("register.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))

@app.route("/predict-page")
@login_required
def predict_page():
    return render_template("index.html", user=current_user, active="predict")

@app.route("/history")
@login_required
def history():
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()
    return render_template("history.html", predictions=predictions, user=current_user, active="history")

@app.route("/dashboard")
@login_required
def dashboard():
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.asc()).all()
    return render_template("dashboard.html", predictions=predictions, user=current_user, active="dashboard")

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    data = request.get_json()

    height = float(data["height"])
    weight = float(data["weight"])
    bmi = round(weight / ((height / 100) ** 2), 2)

    features = [
        float(data["age"]),
        float(data["gender"]),
        height,
        weight,
        float(data["ap_hi"]),
        float(data["ap_lo"]),
        float(data["cholesterol"]),
        float(data["gluc"]),
        float(data["smoke"]),
        float(data["alco"]),
        float(data["active"]),
        bmi
    ]

    input_array = np.array(features).reshape(1, -1)
    probability = model.predict_proba(input_array)[0][1]

    risk_percent = round(probability * 100, 2)

    if risk_percent >= 70:
        risk_level = "High"
    elif risk_percent >= 40:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    # Save prediction to database
    prediction = Prediction(
        user_id=current_user.id,
        age=int(data["age"]),
        gender="Male" if int(data["gender"]) == 2 else "Female",
        height=height,
        weight=weight,
        bmi=bmi,
        ap_hi=int(data["ap_hi"]),
        ap_lo=int(data["ap_lo"]),
        cholesterol=int(data["cholesterol"]),
        gluc=int(data["gluc"]),
        smoke=int(data["smoke"]),
        alco=int(data["alco"]),
        active=int(data["active"]),
        risk_percent=risk_percent,
        risk_level=risk_level
    )
    db.session.add(prediction)
    db.session.commit()

    # Get previous prediction for comparison
    previous = Prediction.query.filter_by(user_id=current_user.id).order_by(
        Prediction.timestamp.desc()).offset(1).first()

    delta = None
    suggestions = []

    if previous:
        delta = round(risk_percent - previous.risk_percent, 2)

        # Smart suggestions based on what changed
        if previous.smoke == 0 and int(data["smoke"]) == 1:
            suggestions.append("You started smoking since last visit — this significantly increases your risk.")
        if previous.smoke == 1 and int(data["smoke"]) == 0:
            suggestions.append("Great job quitting smoking! This has positively impacted your risk score.")
        if float(data["weight"]) > previous.weight + 2:
            suggestions.append(f"Your weight increased by {round(float(data['weight']) - previous.weight, 1)} kg — try to maintain a healthy weight.")
        if float(data["weight"]) < previous.weight - 2:
            suggestions.append(f"You lost {round(previous.weight - float(data['weight']), 1)} kg since last visit — keep it up!")
        if int(data["ap_hi"]) > previous.ap_hi + 10:
            suggestions.append("Your blood pressure has increased — reduce sodium intake and manage stress.")
        if int(data["ap_hi"]) < previous.ap_hi - 10:
            suggestions.append("Your blood pressure improved since last visit — great progress!")
        if previous.active == 0 and int(data["active"]) == 1:
            suggestions.append("You became physically active since last visit — this is helping your heart health!")
        if previous.active == 1 and int(data["active"]) == 0:
            suggestions.append("You stopped being physically active — try to exercise at least 30 mins daily.")
        if previous.alco == 0 and int(data["alco"]) == 1:
            suggestions.append("You started consuming alcohol — moderate or avoid it for better heart health.")

    return jsonify({
        "risk_percent": risk_percent,
        "risk_level": risk_level,
        "bmi": bmi,
        "delta": delta,
        "suggestions": suggestions
    })

@app.route("/clear-history", methods=["POST"])
@login_required
def clear_history():
    Prediction.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    return redirect(url_for("history"))

@app.route("/delete-prediction/<int:prediction_id>", methods=["POST"])
@login_required
def delete_prediction(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)
    if prediction.user_id != current_user.id:
        return "Unauthorized", 403
    db.session.delete(prediction)
    db.session.commit()
    return redirect(url_for("history"))

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
from flask import make_response
import io

@app.route("/report/<int:prediction_id>")
@login_required
def generate_report(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)
    
    if prediction.user_id != current_user.id:
        return "Unauthorized", 403

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                           rightMargin=inch, leftMargin=inch,
                           topMargin=inch, bottomMargin=inch)

    styles = getSampleStyleSheet()
    elements = []

    # Colors
    red = HexColor('#e63946')
    dark = HexColor('#0f172a')
    muted = HexColor('#64748b')
    green = HexColor('#16a34a')
    yellow = HexColor('#d97706')

    # Title
    title_style = ParagraphStyle('title', fontSize=18, fontName='Helvetica-Bold', textColor=dark, spaceAfter=4)
    subtitle_style = ParagraphStyle('subtitle', fontSize=11, fontName='Helvetica',
                                     textColor=muted, spaceAfter=20)
    heading_style = ParagraphStyle('heading', fontSize=13, fontName='Helvetica-Bold',
                                    textColor=dark, spaceBefore=16, spaceAfter=8)
    body_style = ParagraphStyle('body', fontSize=10, fontName='Helvetica',
                                 textColor=muted, spaceAfter=6, leading=16)

    elements.append(Paragraph("HeartGuard", ParagraphStyle('logo', fontSize=14,
                               fontName='Helvetica-Bold', textColor=red, spaceAfter=10)))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("Cardiovascular Risk Assessment Report", title_style))
    elements.append(Spacer(1, 0.05*inch))
    elements.append(Paragraph(f"Generated for {current_user.name} — {prediction.timestamp.strftime('%B %d, %Y at %I:%M %p')}", subtitle_style))
    elements.append(Spacer(1, 0.2*inch))

    # Risk Result
    risk_color = red if prediction.risk_level == "High" else (yellow if prediction.risk_level == "Moderate" else green)
    elements.append(Paragraph("Risk Assessment Result", heading_style))
    
    risk_data = [
        ["Risk Score", "Risk Level", "Assessment Date"],
        [f"{prediction.risk_percent}%", prediction.risk_level, prediction.timestamp.strftime('%d %b %Y')]
    ]
    risk_table = Table(risk_data, colWidths=[2*inch, 2*inch, 2.5*inch])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), HexColor('#f1f5f9')),
        ('TEXTCOLOR', (0,0), (-1,0), muted),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('FONTNAME', (0,1), (-1,1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,1), (-1,1), 13),
        ('TEXTCOLOR', (0,1), (0,1), risk_color),
        ('TEXTCOLOR', (1,1), (1,1), risk_color),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWHEIGHT', (0,0), (-1,-1), 32),
        ('GRID', (0,0), (-1,-1), 1, HexColor('#e2e8f0')),
        ('ROUNDEDCORNERS', [6,6,6,6]),
    ]))
    elements.append(risk_table)
    elements.append(Spacer(1, 0.2*inch))

    # Health Details
    elements.append(Paragraph("Health Parameters", heading_style))
    
    chol_map = {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}
    gluc_map = {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}
    
    health_data = [
        ["Parameter", "Value"],
        ["Age", f"{prediction.age} years"],
        ["Gender", prediction.gender],
        ["Height", f"{prediction.height} cm"],
        ["Weight", f"{prediction.weight} kg"],
        ["BMI", f"{prediction.bmi}"],
        ["Systolic Blood Pressure", f"{prediction.ap_hi} mmHg"],
        ["Diastolic Blood Pressure", f"{prediction.ap_lo} mmHg"],
        ["Cholesterol", chol_map.get(prediction.cholesterol, "Unknown")],
        ["Glucose Level", gluc_map.get(prediction.gluc, "Unknown")],
        ["Smoking", "Yes" if prediction.smoke else "No"],
        ["Alcohol Consumption", "Yes" if prediction.alco else "No"],
        ["Physical Activity", "Yes" if prediction.active else "No"],
    ]

    health_table = Table(health_data, colWidths=[3*inch, 3.5*inch])
    health_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), HexColor('#f1f5f9')),
        ('TEXTCOLOR', (0,0), (-1,0), muted),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,1), (0,-1), dark),
        ('TEXTCOLOR', (1,1), (1,-1), muted),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [HexColor('#ffffff'), HexColor('#f8fafc')]),
        ('GRID', (0,0), (-1,-1), 1, HexColor('#e2e8f0')),
        ('ROWHEIGHT', (0,0), (-1,-1), 26),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
    ]))
    elements.append(health_table)
    elements.append(Spacer(1, 0.2*inch))

    # Recommendations
    elements.append(Paragraph("Personalized Recommendations", heading_style))
    
    tips_map = {
        "Low": [
            "Maintain your current healthy lifestyle.",
            "Exercise regularly — at least 30 minutes a day.",
            "Keep cholesterol and blood pressure in check with regular checkups.",
            "Avoid smoking and limit alcohol intake."
        ],
        "Moderate": [
            "Consult a doctor for a full cardiac checkup.",
            "Reduce saturated fats and processed foods in your diet.",
            "Monitor your blood pressure regularly.",
            "Increase physical activity gradually — aim for 150 mins/week.",
            "Manage stress through meditation or yoga."
        ],
        "High": [
            "Seek medical attention as soon as possible.",
            "Avoid strenuous physical activity until cleared by a doctor.",
            "Strictly monitor blood pressure and cholesterol levels.",
            "Quit smoking immediately if applicable.",
            "Follow a heart-healthy diet — low sodium, low saturated fat."
        ]
    }

    for tip in tips_map.get(prediction.risk_level, []):
        elements.append(Paragraph(f"• {tip}", body_style))

    elements.append(Spacer(1, 0.3*inch))

    # Disclaimer
    disclaimer_style = ParagraphStyle('disclaimer', fontSize=8, fontName='Helvetica',
                                       textColor=HexColor('#94a3b8'), spaceAfter=6,
                                       borderPad=8, leading=14)
    elements.append(Paragraph(
        "⚠ DISCLAIMER: This report is generated by an AI-powered tool for informational purposes only. "
        "It does not constitute medical advice, diagnosis, or treatment. Please consult a qualified "
        "healthcare professional for proper medical evaluation.",
        disclaimer_style
    ))

    doc.build(elements)
    buffer.seek(0)

    response = make_response(buffer.read())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename=HeartGuard_Report_{prediction.timestamp.strftime("%Y%m%d")}.pdf'
    return response
# ─── Create DB ─────────────────────────────────────────────────

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)