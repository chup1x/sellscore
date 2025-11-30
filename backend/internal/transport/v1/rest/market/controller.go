package marketcntrl

import (
	carserv "github.com/chup1x/sellscore/internal/services/market"
	"github.com/go-playground/validator/v10"
	"github.com/gofiber/fiber/v2"
)

type marketAnalysisController struct {
	s         *carserv.CarAnalysisService
	validator *validator.Validate
}

func NewMarketAnalysisController(s *carserv.CarAnalysisService) *marketAnalysisController {
	return &marketAnalysisController{
		s:         s,
		validator: validator.New(),
	}
}

func (m *marketAnalysisController) marketAnalyzeHandler(c *fiber.Ctx) error {
	var req getMarketAnalyzeRequest
	if err := c.BodyParser(&req); err != nil {
		return c.SendStatus(fiber.StatusUnprocessableEntity)
	}
	if err := m.validator.Struct(req); err != nil {
		return c.SendStatus(fiber.StatusUnprocessableEntity)
	}

	carInfo, err := m.s.AnalyzeCar(c.UserContext(), req.CarInfoEntity)
	if err != nil {
		return c.SendStatus(fiber.StatusInternalServerError)
	}
	if _, ok := carInfo["error"]; ok {
		return c.Status(fiber.StatusInternalServerError).JSON(carInfo)
	}

	return c.JSON(carInfo)
}
