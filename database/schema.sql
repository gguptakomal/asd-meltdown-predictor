use asd_predictor;
CREATE TABLE IF NOT EXISTS users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS child_profiles (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,

    child_name VARCHAR(100) NOT NULL,
    age INT NOT NULL,
    gender VARCHAR(20),
    notes TEXT,
    address VARCHAR(255),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id)
    REFERENCES users(id)
    ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS predictions (
    id INT PRIMARY KEY AUTO_INCREMENT,

    user_id INT NOT NULL,
    child_id INT NOT NULL,

    sleep_hours FLOAT,
    stress_level INT,
    social_interaction INT,
    sensory_sensitivity INT,
    anxiety_level INT,
    routine_change BOOLEAN,
    noise_tolerance INT,

    probability FLOAT,
    risk_level VARCHAR(50),
    prediction_result VARCHAR(50),
    notes TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id)
    REFERENCES users(id)
    ON DELETE CASCADE,

    FOREIGN KEY (child_id)
    REFERENCES child_profiles(id)
    ON DELETE CASCADE
);