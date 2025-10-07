package app

import (
	"context"
	"log"

	"github.com/chup1x/sellscore/internal/config"
	"github.com/chup1x/sellscore/internal/transport/v1/rest"
)

func MustRunApp() {
	config, err := config.ReadConfig()
	if err != nil {
		log.Fatalf("read config: %s", err.Error())
	}

	server := rest.New()
	if err := server.Start(context.Background(), config); err != nil {
		log.Fatalf("start web server: %s", err.Error())
	}
}
