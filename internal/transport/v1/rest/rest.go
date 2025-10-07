package rest

import (
	"context"
	"fmt"

	"github.com/chup1x/sellscore/internal/config"
	marketcntrl "github.com/chup1x/sellscore/internal/transport/v1/rest/market"
	"github.com/gofiber/fiber/v2"
)

type Server struct {
	app *fiber.App
}

func New() *Server {
	return &Server{}
}

func (s *Server) Start(_ context.Context, config *config.Config) error {
	s.app = fiber.New()

	api := s.app.Group("/api/v1")

	marketcntrl.RegisterMarketAnalysisRoutes(api, config)

	if err := s.app.Listen(fmt.Sprintf(":%s", config.Server.Port)); err != nil {
		return fmt.Errorf("server start: unable to start web server")
	}

	return nil
}
