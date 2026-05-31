import re
import unicodedata
import os
import json
import shutil
import secrets
from datetime import datetime, timedelta
from flask import Flask, flash, redirect, render_template, request, session, url_for
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()  # Charge les variables du fichier .env

# ── PDF & OCR ──
import fitz          # pymupdf
import pytesseract
from PIL import Image
import io
from pypdf import PdfReader

# ── NLP & ML ──
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# ── Tesseract auto-détection ──
tesseract_path = shutil.which("tesseract")
if not tesseract_path:
    for path in ['/opt/homebrew/bin/tesseract', '/usr/local/bin/tesseract']:
        if os.path.exists(path):
            tesseract_path = path
            break
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    print(f"✅ Tesseract : {tesseract_path}")
else:
    print("⚠️  Tesseract introuvable — OCR désactivé")

# ── SpaCy ──
try:
    import spacy
    nlp = spacy.load("fr_core_news_sm")
    SPACY_OK = True
    print("✅ SpaCy fr_core_news_sm chargé")
except Exception:
    SPACY_OK = False
    print("⚠️  SpaCy non disponible — fallback regex")

# ══════════════════════════════════════════
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "yassir_rh_secret_key_2026")
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)  # durée "Se souvenir de moi"

# ── Config Flask-Mail (Gmail) ──
app.config['MAIL_SERVER']         = 'smtp.gmail.com'
app.config['MAIL_PORT']           = 587
app.config['MAIL_USE_TLS']        = True
app.config['MAIL_USERNAME']       = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD']       = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_USERNAME')

mail = Mail(app)
print("MAIL_USERNAME =", app.config['MAIL_USERNAME'])
print("MAIL_PASSWORD =", "OK" if app.config['MAIL_PASSWORD'] else "VIDE")

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(f):
    return '.' in f and f.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ══════════════════════════════════════════
#   BASE DE DONNÉES
# ══════════════════════════════════════════
def get_db_connection():
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        database=os.environ.get("DB_NAME", "yassir_rh"),
        user=os.environ.get("DB_USER", "asma"),
        password=os.environ.get("DB_PASSWORD", ""),
        cursor_factory=RealDictCursor
    )

# ══════════════════════════════════════════
#   NLP UTILITIES
# ══════════════════════════════════════════
def normalize(s: str) -> str:
    s = (s or "").lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def extract_pdf_text(file_path: str) -> str:
    """Extrait le texte d'un PDF. Fallback OCR si PDF scanné."""
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            text = "\n".join(p.extract_text() or "" for p in reader.pages).strip()
    except Exception as e:
        print(f"pypdf error: {e}")

    # Fallback OCR si texte insuffisant
    if len(text) < 50 and tesseract_path:
        try:
            doc = fitz.open(file_path)
            ocr_pages = []
            for page in doc:
                pix = page.get_pixmap(dpi=150)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_pages.append(pytesseract.image_to_string(img, lang="fra+eng"))
            text = "\n".join(ocr_pages).strip()
            print(f"  → OCR utilisé sur {os.path.basename(file_path)}")
        except Exception as e:
            print(f"OCR error: {e}")
    return text

def extraire_nom_candidat(texte: str, nom_fichier: str) -> str:
    """Tente d'extraire le nom du candidat depuis les premières lignes du CV."""
    lignes = [l.strip() for l in texte.split('\n') if l.strip()]
    for ligne in lignes[:8]:
        mots = ligne.split()
        if 2 <= len(mots) <= 4 and all(m[0].isupper() for m in mots if m.isalpha()):
            if not any(kw in ligne.lower() for kw in
                       ['curriculum', 'vitae', 'cv', 'résumé', 'resume', '@', 'tel', 'email', 'www']):
                return ligne.title()
    return nom_fichier.replace('.pdf', '').replace('_', ' ').replace('-', ' ').title()

# ── TF-IDF cosine ──
def compute_tfidf_cosine(texts: list, query: str) -> np.ndarray:
    corpus = texts + [query]
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    matrix = vec.fit_transform([normalize(t) for t in corpus])
    return cosine_similarity(matrix[-1], matrix[:-1])[0]

# ── KNN boost ──
def compute_knn_boost(scores: np.ndarray, k: int = 5) -> np.ndarray:
    if len(scores) < 2:
        return np.ones(len(scores))
    X = scores.reshape(-1, 1)
    k_eff = min(k, len(X))
    knn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    knn.fit(X)
    distances, _ = knn.kneighbors(X)
    mean_dist = distances.mean(axis=1)
    return 1.0 - (mean_dist / (mean_dist.max() + 1e-9))

# Patterns regex pour expérience / formation / langues
PATTERNS_EXPERIENCE = [
    r'\b(\d+)\s*an[s]?\s*(d\'|de\s*)?(expérience|exp\.?)',
    r'\b(stage|stagiaire|intern)\b',
    r'\b(chef de projet|lead|manager|directeur|responsable)\b',
    r'\b(cdd|cdi|freelance|consultant)\b',
    r'\b(\d{4})\s*[-–]\s*(\d{4}|\bprésent\b|\bactuel\b)',
]
PATTERNS_FORMATION = [
    r'\b(master|licence|bachelor|bts|dut|bac\+\d|ingénieur|doctorat|phd|mba)\b',
    r'\b(université|ecole|école|faculté|institut|ensa|esi|usthb|enp)\b',
    r'\b(diplômé|diplôme|formation|certifi)\b',
]
PATTERNS_LANGUES = [
    r'\b(français|anglais|arabe|espagnol|allemand|italien)\b',
    r'\b(french|english|arabic|spanish|german)\b',
    r'\b(bilingue|trilingue|natif|courant|intermédiaire)\b',
    r'\b(toeic|toefl|ielts|delf|dalf)\b',
    r'\b(b[12]|c[12]|a[12])\b',
]

def _pattern_score(texte: str, patterns: list) -> int:
    t = texte.lower()
    hits = sum(1 for p in patterns if re.search(p, t))
    return min(int((hits / len(patterns)) * 100 * 1.8), 95)

def analyser_competences(texte: str, competences_str: str):
    """Retourne (score, trouvées, manquantes) avec SpaCy si dispo."""
    keywords = [k.strip().lower() for k in competences_str.split(',') if k.strip()]
    if not keywords:
        return 0, [], []

    trouvees, manquantes = [], []

    if SPACY_OK:
        doc_cv = nlp(texte.lower())
        cv_lemmas = {token.lemma_ for token in doc_cv}
        for kw in keywords:
            doc_kw = nlp(kw)
            kw_lemmas = [token.lemma_ for token in doc_kw]
            if all(lm in cv_lemmas for lm in kw_lemmas):
                trouvees.append(kw.title())
            else:
                manquantes.append(kw.title())
    else:
        texte_low = texte.lower()
        for kw in keywords:
            variantes = [
                re.escape(kw),
                re.escape(kw + 's'),
                re.escape(kw.replace(' ', '-')),
                re.escape(kw.replace('-', ' ')),
            ]
            pat = r'\b(' + '|'.join(variantes) + r')\b'
            if re.search(pat, texte_low):
                trouvees.append(kw.title())
            else:
                manquantes.append(kw.title())

    ratio = len(trouvees) / len(keywords)
    score = int(ratio * 85)
    if ratio >= 0.8:
        score = min(score + 10, 100)
    return score, trouvees, manquantes

def generer_points(trouvees, manquantes, score_exp, score_form, score_lang):
    forts, ameliorer = [], []
    if trouvees:
        forts.append(f"Maîtrise de {', '.join(trouvees[:3])}" + (" et plus" if len(trouvees) > 3 else ""))
    if score_form >= 60:
        forts.append("Formation académique solide et adaptée au poste")
    if score_exp >= 60:
        forts.append("Expérience professionnelle significative")
    if score_lang >= 70:
        forts.append("Bonne maîtrise des langues")
    if manquantes:
        ameliorer.append(f"Compétences manquantes : {', '.join(manquantes[:3])}" + (" et autres" if len(manquantes) > 3 else ""))
    if score_exp < 40:
        ameliorer.append("Expérience professionnelle limitée ou non détaillée")
    if score_form < 40:
        ameliorer.append("Formation non précisée ou peu en lien avec le poste")
    if score_lang < 40:
        ameliorer.append("Compétences linguistiques non mentionnées")
    if not forts:
        forts = ["Candidature soumise à évaluation"]
    if not ameliorer:
        ameliorer = ["Continuer à enrichir le CV avec des projets concrets"]
    return forts[:4], ameliorer[:4]

def evaluer_cv_complet(texte: str, offre: dict, tfidf_score: float) -> dict:
    """
    Scoring final combinant :
    - TF-IDF cosine (similarité globale texte/offre)
    - Overlap compétences (SpaCy lemmatisation ou regex)
    - Patterns expérience / formation / langues
    """
    comp_str = offre.get('competences', '') or ''

    # ── Scores par dimension ──
    score_comp, trouvees, manquantes = analyser_competences(texte, comp_str)
    score_exp  = _pattern_score(texte, PATTERNS_EXPERIENCE)
    score_form = _pattern_score(texte, PATTERNS_FORMATION)
    score_lang = _pattern_score(texte, PATTERNS_LANGUES)

    # Bonus formation si correspondance avec l'offre
    formation_cible = offre.get('formation', '') or ''
    if formation_cible:
        for mot in re.split(r'[,\s]+', formation_cible.lower()):
            if mot and len(mot) > 3 and mot in texte.lower():
                score_form = min(score_form + 15, 100)
                break

    # Overlap ratio brut
    keywords = [k.strip() for k in comp_str.split(',') if k.strip()]
    overlap = len(trouvees) / max(len(keywords), 1)

    # ── Score global ──
    # TF-IDF : 35% | Compétences overlap : 30% | Expérience : 20% | Formation : 10% | Langues : 5%
    score_global = int(
        tfidf_score  * 100 * 0.35 +
        overlap      * 100 * 0.30 +
        score_exp         * 0.20 +
        score_form        * 0.10 +
        score_lang        * 0.05
    )
    score_global = max(0, min(score_global, 100))

    if score_global >= 75:   niveau = "Excellent"
    elif score_global >= 60: niveau = "Bon profil"
    elif score_global >= 40: niveau = "Partiel"
    else:                    niveau = "Insuffisant"

    forts, ameliorer = generer_points(trouvees, manquantes, score_exp, score_form, score_lang)

    return {
        'score':                  score_global,
        'score_competences':      score_comp,
        'score_experience':       score_exp,
        'score_formation':        score_form,
        'score_langues':          score_lang,
        'competences_trouvees':   json.dumps(trouvees, ensure_ascii=False),
        'competences_manquantes': json.dumps(manquantes, ensure_ascii=False),
        'points_forts':           json.dumps(forts, ensure_ascii=False),
        'points_ameliorer':       json.dumps(ameliorer, ensure_ascii=False),
        'niveau':                 niveau,
    }

# ══════════════════════════════════════════
#   ROUTES AUTH
# ══════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        nom        = request.form.get('nom')
        entreprise = request.form.get('entreprise')
        email      = request.form.get('email')
        password   = generate_password_hash(request.form.get('password'))
        # Génère un code à 6 chiffres
        code = str(secrets.randbelow(900000) + 100000)

        conn = get_db_connection()
        cur  = conn.cursor()
        try:
            cur.execute(
                """INSERT INTO users (nom, entreprise, email, password, verification_token, is_verified)
                   VALUES (%s, %s, %s, %s, %s, FALSE);""",
                (nom, entreprise, email, password, code))
            conn.commit()

            # Envoie le code par email
            msg = Message("Votre code de vérification", recipients=[email])
            msg.body = (
                f"Bonjour {nom},\n\n"
                f"Votre code de vérification est :\n\n"
                f"  {code}\n\n"
                f"Saisissez ce code sur la page de confirmation pour activer votre compte.\n"
                f"Ce code est valable 24h.\n\n"
                f"Si vous n'êtes pas à l'origine de cette inscription, ignorez cet email."
            )
            mail.send(msg)

            # Stocke l'email en session pour la page de vérification
            session['pending_email'] = email
            flash("Inscription réussie ! Un code à 6 chiffres a été envoyé à votre adresse email.", "success")
            return redirect(url_for('verify_code'))
        except Exception as e:
            conn.rollback()
            print(f"Register error: {e}")
            flash("Cet email est déjà enregistré.", "danger")
        finally:
            cur.close()
            conn.close()
    return render_template('register.html')


# ── Route LOGIN ──
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()
        conn.close()

        # Vérification sécurisée du hash
        if user and check_password_hash(user['password'], password):
            if not user['is_verified']:
                flash("Veuillez d'abord vérifier votre compte.", "warning")
                return redirect(url_for('verify_code'))
            session['user_id'] = user['id']
            return redirect(url_for('dashboard')) # Assurez-vous d'avoir cette route
        
        flash("Email ou mot de passe incorrect.", "danger")
    return render_template('login.html')

# ── Route FORGOT PASSWORD ──
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("SELECT id, nom FROM users WHERE email = %s", (email,))
            user = cur.fetchone()

            if user:
                # Générer code 6 chiffres et expiration (15 min)
                code = str(secrets.randbelow(900000) + 100000)
                expiry = datetime.now() + timedelta(minutes=15)
                cur.execute(
                    "UPDATE users SET reset_code = %s, reset_code_expiry = %s WHERE id = %s",
                    (code, expiry, user['id'])
                )
                conn.commit()

                # Envoyer le code par email
                msg = Message("Réinitialisation de votre mot de passe — Yassir RH", recipients=[email])
                msg.body = (
                    f"Bonjour {user['nom']},\n\n"
                    f"Voici votre code de réinitialisation de mot de passe :\n\n"
                    f"    {code}\n\n"
                    f"Ce code est valable 15 minutes.\n"
                    f"Si vous n'avez pas demandé cette réinitialisation, ignorez cet email.\n\n"
                    f"— L'équipe Yassir RH"
                )
                mail.send(msg)

                # Stocker email ET id en session pour reset_password
                session['reset_user_id'] = user['id']
                session['reset_email']   = email
                flash("Un code de réinitialisation a été envoyé à votre adresse email.", "info")
                return redirect(url_for('reset_password'))

            else:
                # Message neutre (sécurité : ne pas révéler si l'email existe)
                flash("Si cet email est enregistré, vous recevrez un code sous peu.", "info")
                return redirect(url_for('forgot_password'))

        except Exception as e:
            conn.rollback()
            print(f"Forgot password error: {e}")
            flash("Une erreur est survenue. Veuillez réessayer.", "danger")
        finally:
            cur.close()
            conn.close()

    return render_template('forgot_password.html')

# ── Route RESET PASSWORD ──
@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    user_id = session.get('reset_user_id')
    email   = session.get('reset_email', '')

    if not user_id:
        flash("Session expirée. Veuillez recommencer.", "warning")
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        code             = request.form.get('code', '').strip()
        new_password     = request.form.get('password', '')
        confirm_password = request.form.get('confirm', '')

        if new_password != confirm_password:
            flash("Les mots de passe ne correspondent pas.", "danger")
            return render_template('reset_password.html', email=email)

        if len(new_password) < 6:
            flash("Le mot de passe doit contenir au moins 6 caractères.", "danger")
            return render_template('reset_password.html', email=email)

        conn = get_db_connection()
        cur = conn.cursor()
        try:
            # Vérifier code valide + non expiré
            cur.execute("""
                SELECT id FROM users
                WHERE id = %s AND reset_code = %s
                AND reset_code_expiry > NOW()
            """, (user_id, code))
            user = cur.fetchone()

            if user:
                cur.execute("""
                    UPDATE users
                    SET password = %s, reset_code = NULL, reset_code_expiry = NULL
                    WHERE id = %s
                """, (generate_password_hash(new_password), user_id))
                conn.commit()
                # Nettoyer la session
                session.pop('reset_user_id', None)
                session.pop('reset_email', None)
                flash("Mot de passe mis à jour avec succès ! Vous pouvez vous connecter.", "success")
                return redirect(url_for('login'))

            flash("Code invalide ou expiré. Vérifiez le code reçu par email.", "danger")

        except Exception as e:
            conn.rollback()
            print(f"Reset password error: {e}")
            flash("Une erreur est survenue. Veuillez réessayer.", "danger")
        finally:
            cur.close()
            conn.close()

    return render_template('reset_password.html', email=email)


                 
@app.route('/verify-reset-code', methods=['GET', 'POST'])
def verify_reset_code():
    email = session.get('reset_email')

    if not email:
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        code = request.form.get('code', '').strip()

        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT id FROM users
            WHERE email = %s
            AND reset_code = %s
            AND reset_code_expiry > NOW()
        """, (email, code))

        user = cur.fetchone()

        if user:
            session['reset_user_id'] = user['id']
            return redirect(url_for('reset_password'))

        flash("Code incorrect ou expiré", "danger")

    return render_template('verify_reset_code.html')
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

def auth_required():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return None

# ══════════════════════════════════════════
#   DASHBOARD
# ══════════════════════════════════════════
@app.route('/dashboard')
def dashboard():
    r = auth_required()
    if r: return r
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("SELECT * FROM offres WHERE user_id=%s ORDER BY created_at DESC;", (session['user_id'],))
    offres = cur.fetchall()
    cur.execute("SELECT COUNT(*) as n FROM offres WHERE user_id=%s;", (session['user_id'],))
    total_offres = cur.fetchone()['n']
    cur.execute("""SELECT COUNT(*) as n FROM cvs c
                   JOIN offres o ON c.offre_id=o.id WHERE o.user_id=%s;""", (session['user_id'],))
    total_cvs = cur.fetchone()['n']
    cur.execute("""SELECT AVG(c.score) as avg FROM cvs c
                   JOIN offres o ON c.offre_id=o.id WHERE o.user_id=%s;""", (session['user_id'],))
    avg = cur.fetchone()['avg']
    taux_match = round(avg) if avg else 0
    cur.close()
    conn.close()
    return render_template('dashboard.html', offres=offres,
                           total_offres=total_offres, total_cvs=total_cvs, taux_match=taux_match)

# ══════════════════════════════════════════
#   OFFRES
# ══════════════════════════════════════════
@app.route('/offres', methods=['GET', 'POST'])
def offres():
    r = auth_required()
    if r: return r
    conn = get_db_connection()
    cur  = conn.cursor()
    if request.method == 'POST':
        try:
            cur.execute("""INSERT INTO offres (titre,competences,experience,formation,description,user_id)
                           VALUES (%s,%s,%s,%s,%s,%s);""",
                        (request.form.get('titre'), request.form.get('competences'),
                         request.form.get('experience'), request.form.get('formation', ''),
                         request.form.get('description'), session['user_id']))
            conn.commit()
            flash("Offre publiée avec succès !", "success")
        except Exception as e:
            conn.rollback()
            print(f"Offre insert error: {e}")
            flash("Erreur lors de la publication de l'offre.", "danger")
        finally:
            cur.close()
            conn.close()
        return redirect(url_for('offres'))
    cur.execute("SELECT * FROM offres WHERE user_id=%s ORDER BY created_at DESC;", (session['user_id'],))
    mes_offres = cur.fetchall()
    cur.close()
    conn.close()
    return render_template('offres.html', offres=mes_offres)

@app.route('/supprimer-offre/<int:offre_id>', methods=['POST'])
def supprimer_offre(offre_id):
    r = auth_required()
    if r: return r
    conn = get_db_connection()
    cur  = conn.cursor()
    try:
        cur.execute("DELETE FROM offres WHERE id = %s AND user_id = %s;",
                    (offre_id, session['user_id']))
        conn.commit()
        flash("Offre supprimée.", "success")
    except Exception as e:
        conn.rollback()
        print(f"Delete offre error: {e}")
        flash("Erreur lors de la suppression.", "danger")
    finally:
        cur.close()
        conn.close()
    return redirect(url_for('offres'))

# ══════════════════════════════════════════
#   MATCHING IA (cœur de l'application)
# ══════════════════════════════════════════
@app.route('/matching', methods=['GET', 'POST'])
def matching():
    r = auth_required()
    if r: return r

    if request.method == 'POST':
        offre_id     = request.form.get('offre_id')
        files        = [f for f in request.files.getlist('cvFiles')
                        if f and f.filename != '' and allowed_file(f.filename)]
        existing_cvs = request.form.getlist('existing_cvs')

        if not offre_id:
            flash("Veuillez sélectionner une offre.", "danger")
            return redirect(url_for('matching'))
        if not files and not existing_cvs:
            flash("Veuillez sélectionner au moins un candidat ou uploader un CV.", "danger")
            return redirect(url_for('matching'))

        conn = get_db_connection()
        cur  = conn.cursor()
        try:
            cur.execute("SELECT * FROM offres WHERE id=%s AND user_id=%s;",
                        (offre_id, session['user_id']))
            offre = cur.fetchone()
            if not offre:
                flash("Offre introuvable.", "danger")
                return redirect(url_for('matching'))

            offre_dict = dict(offre)
            offer_text = f"{offre_dict.get('titre','')} {offre_dict.get('description','')} {offre_dict.get('competences','')}"

            # ── Étape 1 : Extraire tous les textes ──
            candidats_data = []

            # Nouveaux CVs uploadés
            for file in files:
                filename  = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                texte = extract_pdf_text(file_path)
                if not texte.strip():
                    flash(f"⚠️ {filename} : PDF non lisible (scanné sans OCR). Ignoré.", "warning")
                    continue
                candidats_data.append({'filename': filename, 'texte': texte})

            # CVs existants sélectionnés depuis la liste
            for nom_fichier in existing_cvs:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], nom_fichier)
                if os.path.exists(file_path):
                    texte = extract_pdf_text(file_path)
                    if texte.strip():
                        candidats_data.append({'filename': nom_fichier, 'texte': texte})
                else:
                    cur.execute("SELECT contenu FROM cvs WHERE nom_fichier=%s LIMIT 1;", (nom_fichier,))
                    row = cur.fetchone()
                    if row and row['contenu']:
                        candidats_data.append({'filename': nom_fichier, 'texte': row['contenu']})

            if not candidats_data:
                flash("Aucun texte extrait des CVs fournis.", "danger")
                return redirect(url_for('matching'))

            # ── Étape 2 : TF-IDF cosine sur tous les CVs en même temps ──
            all_texts  = [c['texte'] for c in candidats_data]
            tfidf_sims = compute_tfidf_cosine(all_texts, offer_text)

            # ── Étape 3 : KNN boost de ranking ──
            if len(tfidf_sims) >= 2:
                knn_boost   = compute_knn_boost(tfidf_sims)
                tfidf_final = tfidf_sims * 0.8 + knn_boost * 0.2
            else:
                tfidf_final = tfidf_sims

            # ── Étape 4 : Scoring complet par candidat ──
            resultats_liste = []
            for i, c in enumerate(candidats_data):
                res          = evaluer_cv_complet(c['texte'], offre_dict, float(tfidf_final[i]))
                nom_candidat = extraire_nom_candidat(c['texte'], c['filename'])
                resultats_liste.append({**res, 'filename': c['filename'],
                                        'texte': c['texte'], 'nom_candidat': nom_candidat})

            # ── Étape 5 : Tri final par score ──
            resultats_liste.sort(key=lambda x: x['score'], reverse=True)

            # ── Étape 6 : Sauvegarde en BDD (upsert sur conflit) ──
            for res in resultats_liste:
                cur.execute("""
                    INSERT INTO cvs (
                        nom_fichier, contenu, score,
                        score_competences, score_experience, score_formation, score_langues,
                        competences_trouvees, competences_manquantes,
                        points_forts, points_ameliorer,
                        niveau, nom_candidat, offre_id
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (nom_fichier, offre_id) DO UPDATE SET
                        contenu               = EXCLUDED.contenu,
                        score                 = EXCLUDED.score,
                        score_competences     = EXCLUDED.score_competences,
                        score_experience      = EXCLUDED.score_experience,
                        score_formation       = EXCLUDED.score_formation,
                        score_langues         = EXCLUDED.score_langues,
                        competences_trouvees  = EXCLUDED.competences_trouvees,
                        competences_manquantes= EXCLUDED.competences_manquantes,
                        points_forts          = EXCLUDED.points_forts,
                        points_ameliorer      = EXCLUDED.points_ameliorer,
                        niveau                = EXCLUDED.niveau,
                        nom_candidat          = EXCLUDED.nom_candidat;
                """, (
                    res['filename'], res['texte'][:10000], res['score'],
                    res['score_competences'], res['score_experience'],
                    res['score_formation'],   res['score_langues'],
                    res['competences_trouvees'],   res['competences_manquantes'],
                    res['points_forts'],           res['points_ameliorer'],
                    res['niveau'], res['nom_candidat'], offre_id
                ))
            conn.commit()
            flash(f"{len(resultats_liste)} CV(s) analysé(s) avec succès !", "success")
            return redirect(url_for('resultats', offre_id=offre_id))

        except Exception as e:
            conn.rollback()
            print(f"Matching error: {e}")
            flash("Une erreur est survenue lors de l'analyse.", "danger")
            return redirect(url_for('matching'))
        finally:
            cur.close()
            conn.close()

    # GET
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("SELECT id,titre,competences FROM offres WHERE user_id=%s ORDER BY created_at DESC;",
                (session['user_id'],))
    mes_offres = cur.fetchall()
    cur.execute("""
        SELECT DISTINCT ON (c.nom_fichier) c.nom_fichier, c.nom_candidat, c.score, c.offre_id
        FROM cvs c
        JOIN offres o ON c.offre_id = o.id
        WHERE o.user_id = %s
        ORDER BY c.nom_fichier, c.score DESC;
    """, (session['user_id'],))
    candidats_existants = cur.fetchall()
    cur.close()
    conn.close()
    return render_template('matching.html', offres=mes_offres, candidats_existants=candidats_existants)

# ══════════════════════════════════════════
#   RÉSULTATS
# ══════════════════════════════════════════
@app.route('/resultats/<int:offre_id>')
def resultats(offre_id):
    r = auth_required()
    if r: return r
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("SELECT * FROM offres WHERE id=%s AND user_id=%s;",
                (offre_id, session['user_id']))
    offre = cur.fetchone()
    if not offre:
        cur.close()
        conn.close()
        return "Accès refusé.", 403
    cur.execute("SELECT * FROM cvs WHERE offre_id=%s ORDER BY score DESC;", (offre_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    candidats = []
    for idx, row in enumerate(rows):
        candidats.append({
            'rang':                   idx + 1,
            'nom_candidat':           row['nom_candidat'] or row['nom_fichier'].replace('.pdf', '').replace('_', ' ').title(),
            'fichier':                row['nom_fichier'],
            'global_score':           row['score'],
            'score_competences':      row['score_competences'],
            'score_experience':       row['score_experience'],
            'score_formation':        row['score_formation'],
            'score_langues':          row['score_langues'],
            'competences_trouvees':   json.loads(row['competences_trouvees'] or '[]'),
            'competences_manquantes': json.loads(row['competences_manquantes'] or '[]'),
            'points_forts':           json.loads(row['points_forts'] or '[]'),
            'points_ameliorer':       json.loads(row['points_ameliorer'] or '[]'),
            'niveau':                 row['niveau'],
        })
    return render_template('resultats.html', candidats=candidats, offre=dict(offre))

# ══════════════════════════════════════════
#   CANDIDATS
# ══════════════════════════════════════════
@app.route('/candidats')
def candidats():
    r = auth_required()
    if r: return r
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT c.id, c.nom_fichier, c.nom_candidat, c.score, c.niveau, c.created_at,
               o.titre as offre_titre
        FROM cvs c
        JOIN offres o ON c.offre_id=o.id
        WHERE o.user_id=%s
        ORDER BY c.score DESC;
    """, (session['user_id'],))
    mes_candidats = cur.fetchall()
    cur.close()
    conn.close()
    return render_template('candidats.html', candidats=mes_candidats)

@app.route('/candidats/supprimer/<int:candidat_id>', methods=['POST'])
def supprimer_candidat(candidat_id):
    r = auth_required()
    if r: return r
    conn = get_db_connection()
    cur  = conn.cursor()
    try:
        cur.execute("""DELETE FROM cvs c USING offres o
                       WHERE c.offre_id=o.id AND c.id=%s AND o.user_id=%s;""",
                    (candidat_id, session['user_id']))
        conn.commit()
        flash("Candidat supprimé.", "success")
    except Exception as e:
        conn.rollback()
        print(f"Delete error: {e}")
        flash("Erreur lors de la suppression.", "danger")
    finally:
        cur.close()
        conn.close()
    return redirect(url_for('candidats'))

# ══════════════════════════════════════════
#   VÉRIFICATION PAR CODE À 6 CHIFFRES
# ══════════════════════════════════════════
@app.route('/verify', methods=['GET', 'POST'])
def verify_code():
    # Si l'email n'est pas en session, rediriger vers l'inscription
    email = session.get('pending_email')
    if not email:
        flash("Session expirée. Veuillez vous inscrire à nouveau.", "warning")
        return redirect(url_for('register'))

    if request.method == 'POST':
        code_saisi = request.form.get('code', '').strip()
        conn = get_db_connection()
        cur  = conn.cursor()
        try:
            cur.execute(
                "SELECT id FROM users WHERE email = %s AND verification_token = %s AND is_verified = FALSE;",
                (email, code_saisi)
            )
            user = cur.fetchone()
            if user:
                cur.execute(
                    "UPDATE users SET is_verified = TRUE, verification_token = NULL WHERE id = %s;",
                    (user['id'],)
                )
                conn.commit()
                session.pop('pending_email', None)
                flash("Email vérifié ! Vous pouvez maintenant vous connecter.", "success")
                return redirect(url_for('login'))
            else:
                flash("Code incorrect ou expiré. Vérifiez votre email et réessayez.", "danger")
        except Exception as e:
            conn.rollback()
            print(f"Verify error: {e}")
            flash("Une erreur est survenue.", "danger")
        finally:
            cur.close()
            conn.close()

    return render_template('verify.html', email=email)


@app.route('/resend-code')
def resend_code():
    """Renvoie un nouveau code à 6 chiffres si l'utilisateur ne l'a pas reçu."""
    email = session.get('pending_email')
    if not email:
        return redirect(url_for('register'))

    nouveau_code = str(secrets.randbelow(900000) + 100000)
    conn = get_db_connection()
    cur  = conn.cursor()
    try:
        cur.execute(
            "UPDATE users SET verification_token = %s WHERE email = %s AND is_verified = FALSE;",
            (nouveau_code, email)
        )
        conn.commit()
        msg = Message("Votre nouveau code de vérification", recipients=[email])
        msg.body = (
            f"Votre nouveau code de vérification est :\n\n"
            f"  {nouveau_code}\n\n"
            f"Ce code est valable 24h."
        )
        mail.send(msg)
        flash("Un nouveau code a été envoyé à votre adresse email.", "info")
    except Exception as e:
        conn.rollback()
        print(f"Resend error: {e}")
        flash("Impossible de renvoyer le code.", "danger")
    finally:
        cur.close()
        conn.close()
    return redirect(url_for('verify_code'))



if __name__ == '__main__':
    app.run(debug=True, port=5000)