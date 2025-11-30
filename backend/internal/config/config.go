package config

import (
	"fmt"

	"github.com/caarlos0/env/v6"
	"github.com/joho/godotenv"
)

type Config struct {
	Server     ServerConfig
	CarService CarServiceConfig
}

type ServerConfig struct {
	Port string `env:"SERVER_PORT" envDefault:"80"`
}

type CarServiceConfig struct {
	Host string `env:"CAR_SERVICE_HOST" envDefault:"ml"`
	Port string `env:"CAR_SERVICE_PORT" envDefault:"8000"`
	Path string `env:"CAR_SERVICE_PATH" envDefault:"/api/analyze"`
}

func GetConfig() (*Config, error) {
	_ = godotenv.Load()

	config := &Config{}
	if err := env.Parse(config); err != nil {
		return nil, fmt.Errorf("parse .env file: %w", err)
	}

	return config, nil
}
