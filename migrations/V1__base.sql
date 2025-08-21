+50
-0

-- Base schema migration

PRAGMA foreign_keys = ON;

CREATE TABLE leagues (
    league_id   INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    country     TEXT
);

CREATE TABLE bookmakers (
    bookmaker_id    INTEGER PRIMARY KEY,
    name            TEXT NOT NULL
);

CREATE TABLE matches (
    match_id    INTEGER PRIMARY KEY,
    league_id   INTEGER NOT NULL,
    season      INTEGER NOT NULL,
    date        DATETIME NOT NULL,
    home_team   TEXT NOT NULL,
    away_team   TEXT NOT NULL,
    real_result TEXT CHECK(real_result IN ('1','X','2')),
    FOREIGN KEY (league_id) REFERENCES leagues(league_id)
);

CREATE TABLE odds (
    odds_id       INTEGER PRIMARY KEY,
    match_id      INTEGER NOT NULL,
    bookmaker_id  INTEGER NOT NULL,
    odds_home     REAL NOT NULL,
    odds_draw     REAL NOT NULL,
    odds_away     REAL NOT NULL,
    FOREIGN KEY (match_id) REFERENCES matches(match_id),
    FOREIGN KEY (bookmaker_id) REFERENCES bookmakers(bookmaker_id)
);

CREATE TABLE predictions (
    prediction_id    INTEGER PRIMARY KEY,
    match_id         INTEGER NOT NULL,
    model_name       TEXT NOT NULL,
    prob_home        REAL NOT NULL CHECK(prob_home BETWEEN 0 AND 1),
    prob_draw        REAL NOT NULL CHECK(prob_draw BETWEEN 0 AND 1),
    prob_away        REAL NOT NULL CHECK(prob_away BETWEEN 0 AND 1),
    predicted_result TEXT NOT NULL CHECK(predicted_result IN ('1','X','2')),
    is_correct       BOOLEAN CHECK(is_correct IN (0,1)),
    FOREIGN KEY (match_id) REFERENCES matches(match_id),
    CHECK (prob_home + prob_draw + prob_away = 1)
);
