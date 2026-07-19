package main

import (
	"fmt"
	"example.com/proj/internal/store"
	"example.com/proj/util"
)

func main() {
	s := store.New()
	fmt.Println(util.Greet(s.Name()))
}
