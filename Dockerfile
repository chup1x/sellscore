# build bin container
FROM golang:1.25.1 as build

WORKDIR /app

RUN apt-get update && apt-get install -y git

COPY ./ ./

RUN go mod tidy
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o ./server ./cmd/app

# build container
FROM alpine:latest

COPY --from=build /app/server ./

EXPOSE 80

CMD ["./server"]
