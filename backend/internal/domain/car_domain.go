package domain

type CarInfoEntity struct {
	BrandLabel   string  `json:"brand_label"`
	CarName      string  `json:"car_name"`
	Year         int     `json:"year"`
	Price        int     `json:"price"`
	Mileage      int     `json:"mileage"`
	EngineVolume float64 `json:"engine_volume"`
	PowerHP      int     `json:"power_hp"`
	Transmission string  `json:"transmission"`
	Drive        string  `json:"drive"`
	BodyType     string  `json:"body_type"`
	City         string  `json:"city"`
	Color        string  `json:"color"`
}
