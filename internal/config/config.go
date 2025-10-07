package config

import (
	"fmt"

	"github.com/caarlos0/env/v6"
	"github.com/joho/godotenv"
)

type Config struct {
	Server ServerConfig
}

type ServerConfig struct {
	Port string `env:"SERVER_PORT" envDefault:"80"`
}

func ReadConfig() (*Config, error) {
	_ = godotenv.Load()

	config := &Config{}
	if err := env.Parse(config); err != nil {
		return nil, fmt.Errorf("parse .env file: %w", err)
	}

	return config, nil
}
