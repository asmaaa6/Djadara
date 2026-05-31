DROP TABLE IF EXISTS cvs, offres, users CASCADE;

CREATE TABLE IF NOT EXISTS users (
    id                 SERIAL PRIMARY KEY,
    nom                VARCHAR(100)        NOT NULL,
    entreprise         VARCHAR(100),
    email              VARCHAR(100) UNIQUE NOT NULL,
    password           VARCHAR(255)        NOT NULL,
    verification_token VARCHAR(100),
    is_verified        BOOLEAN DEFAULT FALSE,
    reset_code         VARCHAR(6),
    reset_code_expiry  TIMESTAMP,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS offres (
    id          SERIAL PRIMARY KEY,
    titre       VARCHAR(255) NOT NULL,
    description TEXT,
    competences TEXT         NOT NULL,
    experience  VARCHAR(100),
    formation   VARCHAR(150),
    user_id     INTEGER REFERENCES users(id) ON DELETE CASCADE,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cvs (
    id                     SERIAL PRIMARY KEY,
    nom_fichier            VARCHAR(255) NOT NULL,
    contenu                TEXT,
    -- Score global (0-100)
    score                  SMALLINT CHECK (score >= 0 AND score <= 100),
    -- Scores détaillés par catégorie
    score_competences      SMALLINT DEFAULT 0 CHECK (score_competences >= 0 AND score_competences <= 100),
    score_experience       SMALLINT DEFAULT 0 CHECK (score_experience >= 0 AND score_experience <= 100),
    score_formation        SMALLINT DEFAULT 0 CHECK (score_formation >= 0 AND score_formation <= 100),
    score_langues          SMALLINT DEFAULT 0 CHECK (score_langues >= 0 AND score_langues <= 100),
    -- Résultats JSON
    competences_trouvees   TEXT DEFAULT '',
    competences_manquantes TEXT DEFAULT '',
    points_forts           TEXT DEFAULT '',
    points_ameliorer       TEXT DEFAULT '',
    -- Niveau qualitatif
    niveau                 VARCHAR(50) DEFAULT 'À vérifier',
    -- Nom extrait du candidat
    nom_candidat           VARCHAR(255) DEFAULT '',
    offre_id               INTEGER REFERENCES offres(id) ON DELETE CASCADE,
    created_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Un même CV ne peut être soumis qu'une seule fois par offre
    CONSTRAINT uq_cv_par_offre UNIQUE (nom_fichier, offre_id)
);

CREATE INDEX IF NOT EXISTS idx_offres_user ON offres(user_id);
CREATE INDEX IF NOT EXISTS idx_cvs_offre   ON cvs(offre_id);
CREATE INDEX IF NOT EXISTS idx_cvs_fichier ON cvs(nom_fichier);