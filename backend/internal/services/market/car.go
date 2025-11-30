package carserv

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/chup1x/sellscore/internal/domain"
)

type carAnalysisClient struct {
	host, path, port string
	client           *http.Client
}

func NewCarAnalysisClient(host, port, path string) *carAnalysisClient {
	return &carAnalysisClient{
		client: &http.Client{
			Timeout: 5 * time.Second,
		},
		host: host,
		port: port,
		path: path,
	}
}

func (c *carAnalysisClient) analyzeCar(ctx context.Context, body *domain.CarInfoEntity) (map[string]any, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal json body: %w", err)
	}
	r := bytes.NewReader(data)

	addr := fmt.Sprintf("http://%s:%s%s", c.host, c.port, c.path)

	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, addr, r)
	req.Header.Set("Content-Type", "application/json")

	res, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("to get a response from car service: %w", err)
	}
	defer res.Body.Close()

	resBody := make(map[string]any)

	if err := json.NewDecoder(res.Body).Decode(&resBody); err != nil {
		return nil, fmt.Errorf("to decode a json body: %w", err)
	}

	return resBody, nil
}

type CarAnalysisService struct {
	carServ *carAnalysisClient
}

func NewCarAnalysisService(carServ *carAnalysisClient) *CarAnalysisService {
	return &CarAnalysisService{
		carServ: carServ,
	}
}

func (s *CarAnalysisService) AnalyzeCar(ctx context.Context, carInfo *domain.CarInfoEntity) (map[string]any, error) {
	info, err := s.carServ.analyzeCar(ctx, carInfo)
	if err != nil {
		return nil, fmt.Errorf("to analyze car: %w", err)
	}

	return info, nil
}
