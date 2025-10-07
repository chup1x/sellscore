package marketcntrl

import (
	"github.com/chup1x/sellscore/internal/config"
	"github.com/gofiber/fiber/v2"
)

func RegisterMarketAnalysisRoutes(router fiber.Router, _ *config.Config) {
	marketCntrl := NewMarketAnalysisController()
	market := router.Group("/market")
	market.Post("/analyze", marketCntrl.marketAnalyzeHandler)
}
