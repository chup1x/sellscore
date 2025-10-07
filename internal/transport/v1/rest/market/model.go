package marketcntrl

type getMarketAnalyzeRequest struct {
	Make  string `param:"make" validate:"required"`
	Model string `param:"model" validate:"required"`
	Year  string `param:"year" validate:"required"`
	Price string `param:"price" validate:"required"`
}

type getMarketAnalyzeResponse struct {
	SellabilityScore     int    `json:"sellability_score"`
	SaleSpeedForecast    string `json:"sale_speed_forecast"`
	PriceRecommendations struct {
		FastSale  int `json:"fast_sale"`
		MaxProfit int `json:"max_profit"`
		MarketAvg int `json:"market_average"`
	} `json:"price_recommendations"`
}
