package main

import (
	"log"
	"net/http"
	"os"
)

func main() {
	dir := os.Getenv("STATIC_DIR")
	if dir == "" {
		dir = "./public"
	}

	fs := http.FileServer(http.Dir(dir))
	http.Handle("/", fs)

	addr := os.Getenv("STATIC_ADDR")
	if addr == "" {
		addr = ":8080"
	}

	log.Printf("serving %s on %s", dir, addr)
	log.Fatal(http.ListenAndServe(addr, nil))
}
