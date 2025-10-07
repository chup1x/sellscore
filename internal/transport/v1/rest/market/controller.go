package marketcntrl

import (
	"github.com/go-playground/validator/v10"
	"github.com/gofiber/fiber/v2"
)

type marketAnalysisController struct {
	validator *validator.Validate
}

func NewMarketAnalysisController() *marketAnalysisController {
	return &marketAnalysisController{
		validator: validator.New(),
	}
}

func (c *marketAnalysisController) marketAnalyzeHandler(ctx *fiber.Ctx) error {
	return ctx.SendStatus(fiber.StatusOK)
}
