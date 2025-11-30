package marketcntrl

import (
	"github.com/chup1x/sellscore/internal/config"
	carserv "github.com/chup1x/sellscore/internal/services/market"
	"github.com/gofiber/fiber/v2"
)

func RegisterMarketAnalysisRoutes(router fiber.Router, cfg *config.Config) {
	client := carserv.NewCarAnalysisClient(cfg.CarService.Host, cfg.CarService.Port, cfg.CarService.Path)
	marketCntrl := NewMarketAnalysisController(carserv.NewCarAnalysisService(client))
	router.Post("/analyze", marketCntrl.marketAnalyzeHandler)
}
