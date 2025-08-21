CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id TEXT,
    home_team TEXT,
    away_team TEXT,
    date TEXT,
    status TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS simulations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER REFERENCES matches(id),
    strategy TEXT,
    model TEXT,
    profit REAL DEFAULT 0.0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS odds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER REFERENCES matches(id),
    bookmaker_id TEXT,
    home_win REAL,
    draw REAL,
    away_win REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER REFERENCES matches(id),
    model TEXT,
    p_home REAL,
    p_draw REAL,
    p_away REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
